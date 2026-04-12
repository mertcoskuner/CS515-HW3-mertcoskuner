import os
from dataclasses import replace

import torch

from train import run_training, run_pretrained_training, run_kd_training, run_augmix_training
from test import run_test, run_cifar10c_test, run_pgd_test, run_transfer_test
from gradcam import visualize_gradcam_adversarial, visualize_tsne
from helper import set_seed, build_model, build_parta_model, count_flops
from parameters import get_params
from params.data_params import get_data_params
from params.model_params import get_model_params, get_simplecnn_params, get_resnet_params, get_mobilenet_params
from params.model_training_params import get_model_training_params


def main() -> None:
    """Entry point: parse arguments, build params, and run the selected hw part."""
    args = get_params()
    set_seed(args.seed)
    data_params = get_data_params(args)
    model_params = get_model_params(args)
    model_training_params = get_model_training_params(args)

    if args.hw_part == "PART_A":
        device = torch.device(model_training_params.device)
        all_results = {}
        models_for_test = {}
        stem, ext = os.path.splitext(model_training_params.save_path)

        for option in ["resize", "modify"]:
            option_params = replace(
                model_params,
                option=option,
                input_size=224 if option == "resize" else 32,
            )
            model = build_parta_model(option_params, device)

            if args.mode in ("train", "both"):
                _, results = run_pretrained_training(model, option_params, model_training_params, data_params)
                all_results[option] = results
            elif args.mode == "test":
                save_path = f"{stem}_parta_{option}{ext}"
                model.load_state_dict(torch.load(save_path, map_location=device))

            models_for_test[option] = (model, option_params.input_size)

        if args.mode in ("test", "both"):
            for option, (model, input_size) in models_for_test.items():
                run_test(model, data_params, model_training_params, device,
                         model_name=f"PART_A/{option}", input_size=input_size)

        if args.mode in ("train", "both"):
            print("\n" + "=" * 45)
            print("PART_A — Option Comparison")
            print(f"  {'Option':<10} {'Best Val Acc':>12}")
            print("-" * 45)
            for option, res in all_results.items():
                print(f"  {option:<10} {res['best_val_acc']:>12.4f}")
            print("=" * 45)

    elif args.hw_part == "PART_B":
        device = torch.device(model_training_params.device)
        all_results = {}
        stem, ext = os.path.splitext(model_training_params.save_path)

        simplecnn_params  = get_simplecnn_params(args)
        resnet_params     = get_resnet_params(args)
        mobilenet_params  = get_mobilenet_params(args)

        if args.mode in ("train", "both"):
            # ── Step 1: SimpleCNN baseline (no label smoothing) ───────────────────
            simplecnn_baseline_params = replace(model_training_params, label_smoothing=0.0)
            simplecnn = build_model(simplecnn_params).to(device)
            simplecnn, results = run_training(simplecnn, simplecnn_params, simplecnn_baseline_params, data_params)
            all_results["SimpleCNN"] = results

            # ── Step 2: ResNet — without label smoothing ───────────────────────────
            resnet_no_ls_training_params = replace(model_training_params, label_smoothing=0.0)
            resnet_no_ls = build_model(resnet_params).to(device)
            resnet_no_ls, results = run_training(resnet_no_ls, resnet_params, resnet_no_ls_training_params, data_params)
            all_results["ResNet_no_ls"] = results

            # ── Step 3: ResNet — with label smoothing ──────────────────────────────
            resnet_ls = build_model(resnet_params).to(device)
            resnet_ls, results = run_training(resnet_ls, resnet_params, model_training_params, data_params)
            all_results["ResNet_ls"] = results

            # Pick best ResNet as teacher
            if all_results["ResNet_ls"]["best_val_acc"] >= all_results["ResNet_no_ls"]["best_val_acc"]:
                teacher = resnet_ls
            else:
                teacher = resnet_no_ls

            # ── Step 4: SimpleCNN — Knowledge Distillation ────────────────────────
            simplecnn_kd = build_model(simplecnn_params).to(device)
            simplecnn_kd, results = run_kd_training(
                simplecnn_kd, teacher, simplecnn_params, model_training_params, data_params,
                modified_kd=False,
            )
            all_results["SimpleCNN_KD"] = results

            # ── Step 5: MobileNet — Modified KD ───────────────────────────────────
            mobilenet = build_model(mobilenet_params).to(device)
            mobilenet, results = run_kd_training(
                mobilenet, teacher, mobilenet_params, model_training_params, data_params,
                modified_kd=True,
            )
            all_results["MobileNet_KD"] = results

        elif args.mode == "test":
            # ── Load all models from saved checkpoints ─────────────────────────────
            simplecnn = build_model(simplecnn_params).to(device)
            simplecnn.load_state_dict(torch.load(f"{stem}_simplecnn{ext}", map_location=device))

            resnet_no_ls = build_model(resnet_params).to(device)
            resnet_no_ls.load_state_dict(torch.load(f"{stem}_resnet{ext}", map_location=device))

            ls_val = model_training_params.label_smoothing
            ls_tag = f"_ls{ls_val}" if ls_val > 0.0 else ""
            resnet_ls = build_model(resnet_params).to(device)
            resnet_ls.load_state_dict(torch.load(f"{stem}_resnet{ls_tag}{ext}", map_location=device))

            simplecnn_kd = build_model(simplecnn_params).to(device)
            simplecnn_kd.load_state_dict(torch.load(f"{stem}_simplecnn_kd{ext}", map_location=device))

            mobilenet = build_model(mobilenet_params).to(device)
            mobilenet.load_state_dict(torch.load(f"{stem}_mobilenet_modified_kd{ext}", map_location=device))

            teacher = resnet_ls  # same architecture as resnet_no_ls — FLOPs identical

        if args.mode in ("test", "both"):
            for name, model in {
                "SimpleCNN":    simplecnn,
                "ResNet_no_ls": resnet_no_ls,
                "ResNet_ls":    resnet_ls,
                "SimpleCNN_KD": simplecnn_kd,
                "MobileNet_KD": mobilenet,
            }.items():
                run_test(model, data_params, model_training_params, device, model_name=name)

        # ── FLOPs comparison ───────────────────────────────────────────────────────
        input_size = (3, 32, 32)
        flops = {
            "ResNet":       count_flops(teacher,      input_size, device),
            "SimpleCNN_KD": count_flops(simplecnn_kd, input_size, device),
            "MobileNet_KD": count_flops(mobilenet,    input_size, device),
        }

        if args.mode in ("train", "both"):
            print("\n" + "=" * 60)
            print("PART_B — Accuracy Comparison")
            print(f"  {'Model':<20} {'Best Val Acc':>12}")
            print("-" * 60)
            for name, res in all_results.items():
                print(f"  {name:<20} {res['best_val_acc']:>12.4f}")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("PART_B — FLOPs Comparison")
        print(f"  {'Model':<20} {'MACs':>15}")
        print("-" * 60)
        for name, flop_count in flops.items():
            print(f"  {name:<20} {flop_count:>15,}")
        print("=" * 60)

    elif args.hw_part == "PART_C":
        device = torch.device(model_training_params.device)
        # stem/ext for PART_C AugMix + KD checkpoint saving
        stem, ext = os.path.splitext(model_training_params.save_path)
        # stems for loading original PART_A / PART_B checkpoints
        parta_stem, parta_ext = os.path.splitext(args.parta_save_path)
        partb_stem, partb_ext = os.path.splitext(args.partb_save_path)

        simplecnn_params = get_simplecnn_params(args)
        resnet_params    = get_resnet_params(args)
        mobilenet_params = get_mobilenet_params(args)
        ls_val = model_training_params.label_smoothing
        ls_tag = f"_ls{ls_val}" if ls_val > 0.0 else ""

        augmix_kwargs = dict(
            augmix_severity = args.augmix_severity,
            augmix_width    = args.augmix_width,
            augmix_depth    = args.augmix_depth,
            lambda_jsd      = args.lambda_jsd,
        )

        def _load_parta_models():
            """Load original PART_A fine-tuned models (resize + modify)."""
            parta = {}
            for option in ["resize", "modify"]:
                option_params = replace(
                    model_params,
                    option=option,
                    input_size=224 if option == "resize" else 32,
                )
                m = build_parta_model(option_params, device)
                m.load_state_dict(torch.load(f"{parta_stem}_parta_{option}{parta_ext}", map_location=device))
                parta[f"PART_A/{option}"] = (m, option_params.input_size)
            return parta

        def _load_augmix_parta_models():
            """Load PART_A models re-fine-tuned with AugMix."""
            parta_am = {}
            for option in ["resize", "modify"]:
                option_params = replace(
                    model_params,
                    option=option,
                    input_size=224 if option == "resize" else 32,
                )
                m = build_parta_model(option_params, device)
                m.load_state_dict(torch.load(f"{stem}_parta_{option}_augmix{ext}", map_location=device))
                parta_am[f"PART_A/{option}_AugMix"] = (m, option_params.input_size)
            return parta_am

        def _load_augmix_resnet():
            """Load AugMix-trained ResNet models (used as teacher in Steps 4/5)."""
            resnet_no_ls_am = build_model(resnet_params).to(device)
            resnet_no_ls_am.load_state_dict(torch.load(f"{stem}_resnet_augmix{ext}", map_location=device))
            resnet_ls_am = build_model(resnet_params).to(device)
            resnet_ls_am.load_state_dict(torch.load(f"{stem}_resnet{ls_tag}_augmix{ext}", map_location=device))
            return resnet_no_ls_am, resnet_ls_am

        def _load_original_kd_models():
            """Load original KD student models from PART_B."""
            kd = {}
            simplecnn_kd = build_model(simplecnn_params).to(device)
            simplecnn_kd.load_state_dict(torch.load(f"{partb_stem}_simplecnn_kd{partb_ext}", map_location=device))
            kd["SimpleCNN_KD"] = (simplecnn_kd, 32)
            mobilenet = build_model(mobilenet_params).to(device)
            mobilenet.load_state_dict(torch.load(f"{partb_stem}_mobilenet_modified_kd{partb_ext}", map_location=device))
            kd["MobileNet_KD"] = (mobilenet, 32)
            return kd

        def _load_augmix_kd_models():
            """Load AugMix-teacher KD student models from saved checkpoints."""
            am_kd = {}
            simplecnn_am_kd = build_model(simplecnn_params).to(device)
            simplecnn_am_kd.load_state_dict(
                torch.load(f"{stem}_simplecnn_augmix_kd{ext}", map_location=device)
            )
            am_kd["SimpleCNN_AugMix_KD"] = (simplecnn_am_kd, 32)

            mobilenet_am_kd = build_model(mobilenet_params).to(device)
            mobilenet_am_kd.load_state_dict(
                torch.load(f"{stem}_mobilenet_augmix_modified_kd{ext}", map_location=device)
            )
            am_kd["MobileNet_AugMix_KD"] = (mobilenet_am_kd, 32)
            return am_kd

        def _evaluate(models_to_eval):
            """Return summary dict with clean + CIFAR-10-C results for each model."""
            summary = {}
            for name, (model, input_sz) in models_to_eval.items():
                clean_acc = run_test(
                    model, data_params, model_training_params, device,
                    model_name=name, input_size=input_sz,
                )
                c10c_res = run_cifar10c_test(
                    model, data_params, args.cifar10c_dir,
                    model_training_params, device,
                    model_name=name, input_size=input_sz,
                )
                summary[name] = {
                    "clean":        clean_acc,
                    "mean_per_sev": c10c_res["mean_per_severity"],
                    "mean_c":       c10c_res["overall"],
                }
            return summary

        def _print_summary(summary, title):
            print("\n" + "=" * 85)
            print(title)
            print(f"  {'Model':<25} {'Clean':>8}" + "".join(f"  Sev{s}" for s in range(1, 6)) + f"  {'Mean-C':>8}")
            print("-" * 85)
            for name, res in summary.items():
                row = f"  {name:<25} {res['clean']:>8.4f}"
                for sev in range(1, 6):
                    row += f"  {res['mean_per_sev'][sev]:.4f}"
                row += f"  {res['mean_c']:>8.4f}"
                print(row)
            print("=" * 85)

        # ── STEP 1: evaluate original PART_A fine-tuned models ───────────────────
        if args.mode in ("test", "both"):
            print("\n[PART_C — Step 1] Evaluating original fine-tuned models on clean + CIFAR-10-C ...")
            orig_summary = _evaluate(_load_parta_models())
            _print_summary(orig_summary, "PART_C Step 1 — Original Fine-tuned Models: Clean vs. Corrupted")

        # ── STEP 2 training: re-fine-tune PART_A + ResNet with AugMix ────────────
        if args.mode in ("train", "both"):
            print("\n[PART_C — Step 2] Re-training PART_A + ResNet with AugMix ...")

            resnet_no_ls_training = replace(model_training_params, label_smoothing=0.0)

            # PART_A (both options)
            for option in ["resize", "modify"]:
                option_params = replace(
                    model_params,
                    option=option,
                    input_size=224 if option == "resize" else 32,
                )
                m = build_parta_model(option_params, device)
                run_augmix_training(
                    m, option_params, model_training_params, data_params,
                    **augmix_kwargs,
                    input_size=option_params.input_size,
                    save_tag=f"parta_{option}",
                )

            # ResNet (teacher for Step 4)
            resnet_no_ls_am = build_model(resnet_params).to(device)
            resnet_no_ls_am, res_no_ls_am = run_augmix_training(
                resnet_no_ls_am, resnet_params, resnet_no_ls_training, data_params, **augmix_kwargs,
            )

            resnet_ls_am = build_model(resnet_params).to(device)
            resnet_ls_am, res_ls_am = run_augmix_training(
                resnet_ls_am, resnet_params, model_training_params, data_params, **augmix_kwargs,
            )

        # ── STEP 2 evaluation: evaluate AugMix PART_A models ─────────────────────
        if args.mode in ("train", "both"):
            print("\n[PART_C — Step 2] Evaluating AugMix fine-tuned models on clean + CIFAR-10-C ...")
            am_summary = _evaluate(_load_augmix_parta_models())
            _print_summary(am_summary, "PART_C Step 2 — AugMix Fine-tuned Models: Clean vs. Corrupted")

        # ── STEP 3: PGD adversarial robustness + GradCAM + t-SNE ──────────────
        if args.mode in ("test", "both"):
            print("\n[PART_C — Step 3] Adversarial robustness evaluation (PGD-20) ...")

            # Both fine-tuned models: original PART_A + AugMix PART_A
            all_models = {**_load_parta_models(), **_load_augmix_parta_models()}

            adv_summary = {}
            viz_done    = False   # GradCAM + t-SNE only for the first model with samples

            for name, (model, input_sz) in all_models.items():
                need_samples = not viz_done
                res = run_pgd_test(
                    model, data_params, model_training_params, device,
                    model_name=name, input_size=input_sz,
                    collect_samples=need_samples,
                )
                adv_summary[name] = res

                if need_samples and "samples" in res:
                    clean_b, adv_linf_b, adv_l2_b, lbl_b = res["samples"]
                    os.makedirs("plots/partc", exist_ok=True)

                    # GradCAM — L∞ adversarial
                    visualize_gradcam_adversarial(
                        model, clean_b, adv_linf_b, lbl_b, device,
                        save_path=f"plots/partc/gradcam_{name.replace('/', '_')}_linf.png",
                    )
                    # GradCAM — L2 adversarial
                    visualize_gradcam_adversarial(
                        model, clean_b, adv_l2_b, lbl_b, device,
                        save_path=f"plots/partc/gradcam_{name.replace('/', '_')}_l2.png",
                    )
                    # t-SNE — L∞ adversarial
                    visualize_tsne(
                        model, clean_b, adv_linf_b, lbl_b, device,
                        save_path=f"plots/partc/tsne_{name.replace('/', '_')}_linf.png",
                    )
                    viz_done = True

            # PGD summary table
            print("\n" + "=" * 75)
            print("PART_C Step 3 — Adversarial Robustness (PGD-20)")
            print(f"  {'Model':<25} {'Clean':>8}  {'L∞ acc':>8}  {'L2 acc':>8}")
            print("-" * 75)
            for name, res in adv_summary.items():
                print(f"  {name:<25} {res['clean_acc']:>8.4f}  {res['linf_acc']:>8.4f}  {res['l2_acc']:>8.4f}")
            print("=" * 75)

        # ── STEP 4: KD with AugMix teacher (train or both) ────────────────────
        if args.mode in ("train", "both"):
            print("\n[PART_C — Step 4] Training KD students with AugMix teacher ...")

            # Pick best AugMix ResNet as teacher
            augmix_teacher = (
                resnet_ls_am
                if res_ls_am["best_val_acc"] >= res_no_ls_am["best_val_acc"]
                else resnet_no_ls_am
            )

            simplecnn_am_kd = build_model(simplecnn_params).to(device)
            simplecnn_am_kd, _ = run_kd_training(
                simplecnn_am_kd, augmix_teacher, simplecnn_params,
                model_training_params, data_params,
                modified_kd=False, save_tag="simplecnn_augmix",
            )

            mobilenet_am_kd = build_model(mobilenet_params).to(device)
            mobilenet_am_kd, _ = run_kd_training(
                mobilenet_am_kd, augmix_teacher, mobilenet_params,
                model_training_params, data_params,
                modified_kd=True, save_tag="mobilenet_augmix",
            )

            kd_comparison = {**_evaluate(_load_original_kd_models()), **_evaluate(_load_augmix_kd_models())}
            _print_summary(kd_comparison, "PART_C Step 4 — KD: Original Teacher vs. AugMix Teacher")

        # ── STEP 4: evaluate only (test mode) ─────────────────────────────────
        if args.mode == "test":
            print("\n[PART_C — Step 4] Evaluating KD models: original teacher vs. AugMix teacher ...")
            kd_comparison = {**_evaluate(_load_original_kd_models()), **_evaluate(_load_augmix_kd_models())}
            _print_summary(kd_comparison, "PART_C Step 4 — KD: Original Teacher vs. AugMix Teacher")

        # ── STEP 5: Adversarial transferability (test and both modes) ─────────
        if args.mode in ("test", "both"):
            print("\n[PART_C — Step 5] Adversarial transferability: teacher → student ...")

            orig_kd  = _load_original_kd_models()
            am_kd    = _load_augmix_kd_models()

            # Original PART_B teacher (ResNet_ls)
            resnet_ls_orig = build_model(resnet_params).to(device)
            resnet_ls_orig.load_state_dict(
                torch.load(f"{partb_stem}_resnet{ls_tag}{partb_ext}", map_location=device)
            )

            # AugMix teacher (best of ls / no_ls)
            _, resnet_ls_am = _load_augmix_resnet()

            pairs = [
                ("ResNet_ls",        resnet_ls_orig,  "SimpleCNN_KD",        orig_kd["SimpleCNN_KD"][0]),
                ("ResNet_ls",        resnet_ls_orig,  "MobileNet_KD",        orig_kd["MobileNet_KD"][0]),
                ("ResNet_ls_AugMix", resnet_ls_am,    "SimpleCNN_AugMix_KD", am_kd["SimpleCNN_AugMix_KD"][0]),
                ("ResNet_ls_AugMix", resnet_ls_am,    "MobileNet_AugMix_KD", am_kd["MobileNet_AugMix_KD"][0]),
            ]

            transfer_results = {}
            for src_name, src_model, tgt_name, tgt_model in pairs:
                res = run_transfer_test(
                    src_model, tgt_model, data_params, model_training_params, device,
                    source_name=src_name, target_name=tgt_name,
                )
                transfer_results[f"{src_name} → {tgt_name}"] = res

            print("\n" + "=" * 90)
            print("PART_C Step 5 — Adversarial Transferability (PGD-20 L∞ ε=4/255)")
            print(f"  {'Attack pair':<42} {'Src clean':>10}  {'Src adv':>8}  {'Tgt clean':>10}  {'Tgt adv':>8}")
            print("-" * 90)
            for pair, res in transfer_results.items():
                print(f"  {pair:<42} {res['source_clean_acc']:>10.4f}  "
                      f"{res['source_adv_acc']:>8.4f}  "
                      f"{res['target_clean_acc']:>10.4f}  "
                      f"{res['target_adv_acc']:>8.4f}")
            print("=" * 90)

    else:
        raise ValueError(f"Unknown hw part: {args.hw_part}")


if __name__ == "__main__":
    main()
