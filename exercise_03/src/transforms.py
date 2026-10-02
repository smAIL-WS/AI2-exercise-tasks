"""
Exercise 03 - Transform Pipelines
===================================
Define separate transform pipelines for training and validation.

Training transforms include random augmentations to improve generalization.
Validation transforms are deterministic — no randomness.

Run this script to verify:
    python src/transforms.py
"""

from torchvision import transforms


def get_train_transforms(image_size, augmentation=True):
    """
    Returns the transform pipeline for training data.

    Args:
        image_size (int): Target image size (square).
        augmentation (bool): Whether to apply random augmentations.

    Returns:
        torchvision.transforms.Compose
    """
    if augmentation:
        # TODO: Create a Compose pipeline with augmentation.
        #
        #   Step 1: RandomResizedCrop(image_size) — randomly crop a region
        #           and resize it to image_size. This provides both scale
        #           and position variation.
        #   Step 2: RandomHorizontalFlip() — flip with 50% probability.
        #   Step 3: ColorJitter — randomly adjust brightness, contrast,
        #           saturation (0.2 each) and hue (0.1).
        #   Step 4: ToTensor() — converts PIL image to a float tensor
        #           and scales pixel values from [0, 255] to [0, 1].
        #   Step 5: Normalize with ImageNet mean and std.
        #           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        transform = None
    else:
        # TODO: Create a Compose pipeline WITHOUT augmentation.
        #
        #   Step 1: Resize(image_size + 32) — resize to slightly larger.
        #   Step 2: CenterCrop(image_size) — crop the center to exact size.
        #           Resize + CenterCrop ensures consistent framing without
        #           random variation.
        #   Step 3: ToTensor()
        #   Step 4: Normalize with same ImageNet mean and std.
        transform = None

    return transform


def get_val_transforms(image_size):
    """
    Returns the transform pipeline for validation/test data.
    No random augmentations — must be deterministic.

    Args:
        image_size (int): Target image size (square).

    Returns:
        torchvision.transforms.Compose
    """
    # TODO: Same as the non-augmented training pipeline above.
    #       Resize → CenterCrop → ToTensor → Normalize.
    #       Validation must always produce identical output for the
    #       same input, so no random transforms.
    transform = None

    return transform


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import numpy as np
    import torch
    from PIL import Image

    print("=== VERIFICATION ===")

    def _assert_true(condition, message):
        assert condition, message


    def check(label, assertion_fn):
        try:
            assertion_fn()
        except AssertionError as e:
            print(f"[FAIL] {label} - {e}")
            return False
        except NameError as e:
            print(f"[FAIL] {label} (variable not defined: {e})")
            return False
        except Exception as e:
            print(f"[FAIL] {label} (error while checking: {e})")
            return False
        print(f"[PASS] {label}")
        return True


    image_size = 128
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    color = (100, 150, 200)
    # Non-square on purpose: Resize + CenterCrop must still give a square output.
    dummy_img = Image.new("RGB", (300, 200), color=color)
    noise_img = Image.fromarray(np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8))
    # A single-colour image stays single-coloured through resize/crop, so after
    # ToTensor + Normalize every pixel must equal (color / 255 - mean) / std.
    expected_pixel = torch.tensor([(c / 255 - m) / s for c, m, s in zip(color, mean, std)])

    AUG_STEPS = ["RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter", "ToTensor", "Normalize"]
    NO_AUG_STEPS = ["Resize", "CenterCrop", "ToTensor", "Normalize"]


    def _steps(tf, fn_name):
        _assert_true(tf is not None, f"{fn_name} returned None - did you build and return the Compose?")
        _assert_true(isinstance(tf, transforms.Compose),
                     f"expected {fn_name} to return a transforms.Compose, got {type(tf).__name__}")
        return {type(t).__name__: t for t in tf.transforms}, [type(t).__name__ for t in tf.transforms]


    def _check_steps(tf, fn_name, expected):
        steps = _steps(tf, fn_name)[1]
        _assert_true(steps == expected, f"expected steps {expected}, got {steps}")


    def _check_normalize(tf, fn_name):
        step = _steps(tf, fn_name)[0]["Normalize"]
        _assert_true(list(step.mean) == mean and list(step.std) == std,
                     f"expected Normalize(mean={mean}, std={std}), got mean={step.mean}, std={step.std}")


    def _check_resize_crop(tf, fn_name):
        steps = _steps(tf, fn_name)[0]
        resize_size = steps["Resize"].size
        resize_size = resize_size[0] if isinstance(resize_size, (list, tuple)) and len(resize_size) == 1 else resize_size
        _assert_true(resize_size == image_size + 32,
                     f"expected Resize({image_size + 32}) (a single int keeps the aspect ratio), got Resize({resize_size})")
        crop_size = tuple(steps["CenterCrop"].size)
        _assert_true(crop_size == (image_size, image_size),
                     f"expected CenterCrop({image_size}), got CenterCrop({crop_size})")


    def _check_output(tf, fn_name, check_values):
        _steps(tf, fn_name)
        out = tf(dummy_img)
        _assert_true(isinstance(out, torch.Tensor) and out.dtype == torch.float32,
                     f"expected a float32 tensor, got {type(out).__name__} - is ToTensor() in the pipeline?")
        _assert_true(out.shape == torch.Size([3, image_size, image_size]),
                     f"expected shape (3, {image_size}, {image_size}), got {tuple(out.shape)}")
        if check_values:
            torch.testing.assert_close(out.mean(dim=(1, 2)), expected_pixel, atol=1e-2, rtol=0,
                                       msg="pixel values are off - check ToTensor() and Normalize()")


    def _check_aug_params():
        steps = _steps(get_train_transforms(image_size, augmentation=True), "get_train_transforms")[0]
        crop_size = tuple(steps["RandomResizedCrop"].size)
        _assert_true(crop_size == (image_size, image_size),
                     f"expected RandomResizedCrop({image_size}), got size {crop_size}")
        jitter = steps["ColorJitter"]
        got = (jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
        want = ((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        _assert_true(all(g is not None and np.allclose(g, w) for g, w in zip(got, want)),
                     "expected ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)")


    def _check_val_deterministic():
        tf = get_val_transforms(image_size)
        _steps(tf, "get_val_transforms")
        _assert_true(torch.equal(tf(noise_img), tf(noise_img)),
                     "the same image gave two different outputs - remove random transforms from validation")


    def _check_val_matches_no_aug():
        val_tf = get_val_transforms(image_size)
        no_aug_tf = get_train_transforms(image_size, augmentation=False)
        _steps(val_tf, "get_val_transforms")
        _steps(no_aug_tf, "get_train_transforms")
        val_out, no_aug_out = val_tf(noise_img), no_aug_tf(noise_img)
        _assert_true(torch.equal(val_out, no_aug_out),
                     "expected get_val_transforms to match get_train_transforms(augmentation=False)")


    results = []

    results.append(check(
        "get_train_transforms(augmentation=True) - steps are " + " -> ".join(AUG_STEPS),
        lambda: _check_steps(get_train_transforms(image_size, augmentation=True), "get_train_transforms", AUG_STEPS)))
    results.append(check(
        "get_train_transforms(augmentation=True) - RandomResizedCrop and ColorJitter use the given parameters",
        _check_aug_params))
    results.append(check(
        "get_train_transforms(augmentation=True) - normalizes with ImageNet mean/std",
        lambda: _check_normalize(get_train_transforms(image_size, augmentation=True), "get_train_transforms")))
    results.append(check(
        f"get_train_transforms(augmentation=True) - output is a float tensor of shape (3, {image_size}, {image_size})",
        lambda: _check_output(get_train_transforms(image_size, augmentation=True), "get_train_transforms",
                              check_values=False)))

    results.append(check(
        "get_train_transforms(augmentation=False) - steps are " + " -> ".join(NO_AUG_STEPS),
        lambda: _check_steps(get_train_transforms(image_size, augmentation=False), "get_train_transforms",
                             NO_AUG_STEPS)))
    results.append(check(
        "get_train_transforms(augmentation=False) - Resize(image_size + 32) and CenterCrop(image_size)",
        lambda: _check_resize_crop(get_train_transforms(image_size, augmentation=False), "get_train_transforms")))
    results.append(check(
        "get_train_transforms(augmentation=False) - output shape and normalized pixel values",
        lambda: _check_output(get_train_transforms(image_size, augmentation=False), "get_train_transforms",
                              check_values=True)))

    results.append(check(
        "get_val_transforms - steps are " + " -> ".join(NO_AUG_STEPS),
        lambda: _check_steps(get_val_transforms(image_size), "get_val_transforms", NO_AUG_STEPS)))
    results.append(check(
        "get_val_transforms - Resize(image_size + 32) and CenterCrop(image_size)",
        lambda: _check_resize_crop(get_val_transforms(image_size), "get_val_transforms")))
    results.append(check(
        "get_val_transforms - output shape and normalized pixel values",
        lambda: _check_output(get_val_transforms(image_size), "get_val_transforms", check_values=True)))
    results.append(check("get_val_transforms - deterministic (same input gives the same output)",
                         _check_val_deterministic))
    results.append(check("get_val_transforms - identical to get_train_transforms(augmentation=False)",
                         _check_val_matches_no_aug))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
