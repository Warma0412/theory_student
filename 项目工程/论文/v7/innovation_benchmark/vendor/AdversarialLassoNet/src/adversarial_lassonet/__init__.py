"""Public package entry points for adversarial_lassonet."""

__all__ = [
    "AdversarialLassoNetClassifier",
    "AdversarialLassoNetConfig",
    "FeatureSelectionRunResult",
    "LassoNetSAMAligned",
    "AdvLassoNetConfig",
    "RunResult",
    "build_adv_arg_parser",
    "build_adversarial_lassonet",
    "build_improved_adv_lassonet",
    "build_robust_feature_selection_lassonet",
    "canonical_dataset_name",
    "namespace_to_adversarial_lassonet_config",
    "namespace_to_adv_config",
    "namespace_to_robust_lassonet_config",
    "resolve_datasets",
    "resolve_device",
    "save_sparse_checkpoint",
    "set_seed",
    "train_adversarial_lassonet_once",
    "train_improved_adv_lassonet_once",
    "train_robust_feature_selection_lassonet_once",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from models.adversarial_lassonet import (
        AdvLassoNetConfig,
        AdversarialLassoNetClassifier,
        AdversarialLassoNetConfig,
        FeatureSelectionRunResult,
        LassoNetSAMAligned,
        RunResult,
        build_adv_arg_parser,
        build_adversarial_lassonet,
        build_improved_adv_lassonet,
        build_robust_feature_selection_lassonet,
        canonical_dataset_name,
        namespace_to_adversarial_lassonet_config,
        namespace_to_adv_config,
        namespace_to_robust_lassonet_config,
        resolve_datasets,
        resolve_device,
        save_sparse_checkpoint,
        set_seed,
        train_adversarial_lassonet_once,
        train_improved_adv_lassonet_once,
        train_robust_feature_selection_lassonet_once,
    )

    exports = {
        "AdvLassoNetConfig": AdvLassoNetConfig,
        "AdversarialLassoNetClassifier": AdversarialLassoNetClassifier,
        "AdversarialLassoNetConfig": AdversarialLassoNetConfig,
        "FeatureSelectionRunResult": FeatureSelectionRunResult,
        "LassoNetSAMAligned": LassoNetSAMAligned,
        "RunResult": RunResult,
        "build_adv_arg_parser": build_adv_arg_parser,
        "build_adversarial_lassonet": build_adversarial_lassonet,
        "build_improved_adv_lassonet": build_improved_adv_lassonet,
        "build_robust_feature_selection_lassonet": build_robust_feature_selection_lassonet,
        "canonical_dataset_name": canonical_dataset_name,
        "namespace_to_adversarial_lassonet_config": namespace_to_adversarial_lassonet_config,
        "namespace_to_adv_config": namespace_to_adv_config,
        "namespace_to_robust_lassonet_config": namespace_to_robust_lassonet_config,
        "resolve_datasets": resolve_datasets,
        "resolve_device": resolve_device,
        "save_sparse_checkpoint": save_sparse_checkpoint,
        "set_seed": set_seed,
        "train_adversarial_lassonet_once": train_adversarial_lassonet_once,
        "train_improved_adv_lassonet_once": train_improved_adv_lassonet_once,
        "train_robust_feature_selection_lassonet_once": train_robust_feature_selection_lassonet_once,
    }
    value = exports[name]
    globals()[name] = value
    return value
