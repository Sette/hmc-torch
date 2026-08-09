"""Dataset registry with plugin support for HMC-Torch.

Provides a central registry where dataset managers are registered by name.
Built-in datasets (arxiv, wos, aapd, rcv1, eurlex, and GoFun ARFF variants)
auto-register on import.  External packages or downstream users can register
custom datasets via entry points or programmatic registration.

---------------------------------------------------------------------
Quick start — register your own dataset
---------------------------------------------------------------------

The framework derives everything it needs — adjacency matrix, level sizes,
edge indices, input/output dimensions — **from your labels**.  Your only
job is to supply the raw data and make sure the labels encode the hierarchy
in a way the framework can parse.

**1. Pick a label format**

Labels must follow a **dot-separated path notation** where each segment is
one level of the hierarchy.  For example, a paper tagged as ``cs.AI`` and
``stat.ML`` belongs to both *Artificial Intelligence* (under *Computer
Science*) and *Machine Learning* (under *Statistics*)::

    root
    ├── cs          (level 1 — area)
    │   └── cs.AI   (level 2 — subcategory)
    └── stat        (level 1 — area)
        └── stat.ML (level 2 — subcategory)

The same convention works for any tree-shaped taxonomy —
``A.A1.B1``, ``medicine.cardiology``, ``physics.optics.lasers``, etc.

**2. Build the hierarchy** — just collect unique labels and call
:func:`build_digraph_from_labels() <hmc.utils.build_digraph_from_labels>`:

    .. code-block:: python

        from hmc.utils import build_digraph_from_labels
        from hmc.data.hierarchy import TreeHierarchy

        # Collect all unique labels from your data
        all_labels = ["cs", "cs.AI", "cs.LG", "stat", "stat.ML"]

        # One-liner: build the child→parent graph from label strings
        g = build_digraph_from_labels(all_labels)

        hierarchy = TreeHierarchy.from_graph(g)

The function infers intermediate nodes, handles duplicates, and supports
custom separators (``sep="/"``) and root names.  That's it — **you only
need your data and your labels**; the framework derives everything else.

At this point ``hierarchy`` already exposes every attribute the pipeline
needs — ``.a``, ``.edge_index``, ``.to_eval``, ``.local_nodes_idx``,
``.level_sizes``, … — all computed from the graph structure.

**3. Encode labels** with a single call:

    .. code-block:: python

        Y_global, Y_local = hierarchy.encode_labels(
            ["cs.AI", "stat.ML cs.LG"]    # one string per sample (space-separated)
        )
        # Y_global: (n_samples, n_nodes) — ancestors automatically activated
        # Y_local:  list of (n_samples, n_level_nodes) per depth level

**4. Wire everything into your manager.**  Every built-in manager follows
the same pattern inside its ``__init__``:

    a. **Load** raw data and collect unique label strings.
    b. **Build hierarchy** with :func:`TreeHierarchy.from_graph`.
    c. **Compute features** (transformer embeddings, hand-crafted vectors, …).
    d. **Encode labels** with :meth:`Hierarchy.encode_labels`.
    e. **Create splits** (train/valid/test).
    f. **Expose** — copy from hierarchy + splits to plain instance attributes
       (``self.input_dim``, ``self.a``, ``self.edge_index``,
       ``self.get_datasets``, …).

The built-in managers are a good reference for a production version:

    ===================  ===========================================  ==============================
    Manager              File                                        Good for…
    ===================  ===========================================  ==============================
    ``ArXivManager``     :mod:`hmc.datasets.arxiv.manager`           Single JSONL + dot-sep labels
    ``WOSManager``       :mod:`hmc.datasets.wos.manager`             Pre-split JSONL + prebuilt hier
    ``HMCDatasetManager``:mod:`hmc.datasets.gofun.manager`           Tabular ARFF + prebuilt hier
    ===================  ===========================================  ==============================

**5. Register** — no need to modify any package source:

    .. code-block:: python

        from hmc.data import DatasetRegistry

        DatasetRegistry.register(
            "my_data",
            lambda **kw: MyManager(**kw),
            defaults={"hidden_dim": 256, "lr": 1e-4, "epochs": 50, "dropout": 0.3},
        )

**Use it** anywhere in the framework:

    .. code-block:: python

        manager = DatasetRegistry.get("my_data", device="cuda", dataset_path="./data")
        train, valid, test = manager.get_datasets()

        import hmc
        hmc.train(dataset_name="my_data", method="globalE2E", device="cuda", epochs=50)

---------------------------------------------------------------------
Entry points (for external packages)
---------------------------------------------------------------------

If you ship your manager as a pip-installable package, register it
automatically via ``pyproject.toml`` so users don't need to call
:meth:`DatasetRegistry.register` themselves:

.. code-block:: toml

    [project.entry-points."hmc_torch.datasets"]
    my_data = "my_package.manager:create_manager"

The entry-point target (``create_manager`` above) must be a callable that
accepts ``**kwargs`` and returns a manager instance — exactly like the
factory you'd pass to :meth:`DatasetRegistry.register`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar
from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset default hyperparameters (carried over from datasets/registry.py)
# ---------------------------------------------------------------------------


@dataclass
class _HyperDefaults:
    """Internal lookup for default hyperparameters per dataset family."""

    input_dims: dict[str, int] = field(
        default_factory=lambda: {
            "arxiv": 768,
            "wos": 768,
            "aapd": 768,
            "rcv1": 768,
            "eurlex": 768,
            "diatoms": 371,
            "enron": 1001,
            "imclef07a": 80,
            "imclef07d": 80,
            "cellcycle": 77,
            "church": 31,
            "derisi": 63,
            "eisen": 79,
            "expr": 561,
            "gasch1": 173,
            "gasch2": 52,
            "pheno": 276,
            "seq": 529,
            "spo": 86,
        }
    )

    arxiv_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 50,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )
    wos_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 50,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )
    aapd_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 50,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )
    rcv1_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 50,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )
    eurlex_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 30,
            "weight_decay": 1e-5,
            "batch_size": 16,
            "num_layers": 2,
            "dropout": 0.3,
        }
    )
    gofun_defaults: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "lr": 1e-4,
            "epochs": 100,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_layers": 3,
            "dropout": 0.3,
        }
    )

    output_dims: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 499,
                "church": 499,
                "derisi": 499,
                "eisen": 461,
                "expr": 499,
                "gasch1": 499,
                "gasch2": 499,
                "pheno": 455,
                "seq": 499,
                "spo": 499,
            },
            "GO": {
                "cellcycle": 4122,
                "derisi": 4116,
                "eisen": 3570,
                "expr": 4128,
                "gasch1": 4122,
                "gasch2": 4128,
                "pheno": 4500,
                "seq": 4130,
                "spo": 4116,
            },
            "others": {"diatoms": 398, "enron": 56, "imclef07a": 96, "imclef07d": 46},
        }
    )

    hidden_dims: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 500,
                "church": 300,
                "derisi": 500,
                "eisen": 500,
                "expr": 1250,
                "gasch1": 1000,
                "gasch2": 500,
                "pheno": 500,
                "seq": 2000,
                "spo": 250,
            },
            "GO": {
                "cellcycle": 1000,
                "derisi": 500,
                "eisen": 500,
                "expr": 4000,
                "gasch1": 500,
                "gasch2": 500,
                "pheno": 500,
                "seq": 9000,
                "spo": 500,
            },
            "others": {
                "diatoms": 2000,
                "enron": 1000,
                "imclef07a": 1000,
                "imclef07d": 1000,
            },
        }
    )

    lrs: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 1e-4,
                "church": 1e-4,
                "derisi": 1e-4,
                "eisen": 1e-4,
                "expr": 1e-4,
                "gasch1": 1e-4,
                "gasch2": 1e-4,
                "pheno": 1e-4,
                "seq": 1e-4,
                "spo": 1e-4,
            },
            "GO": {
                "cellcycle": 1e-4,
                "derisi": 1e-4,
                "eisen": 1e-4,
                "expr": 1e-4,
                "gasch1": 1e-4,
                "gasch2": 1e-4,
                "pheno": 1e-4,
                "seq": 1e-4,
                "spo": 1e-4,
            },
            "others": {
                "diatoms": 1e-5,
                "enron": 1e-5,
                "imclef07a": 1e-5,
                "imclef07d": 1e-5,
            },
        }
    )

    all_epochs: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "FUN": {
                "cellcycle": 106,
                "church": 100,
                "derisi": 67,
                "eisen": 110,
                "expr": 20,
                "gasch1": 42,
                "gasch2": 123,
                "pheno": 100,
                "seq": 13,
                "spo": 115,
            },
            "GO": {
                "cellcycle": 62,
                "derisi": 91,
                "eisen": 123,
                "expr": 70,
                "gasch1": 122,
                "gasch2": 177,
                "pheno": 100,
                "seq": 45,
                "spo": 103,
            },
            "others": {
                "diatoms": 474,
                "enron": 133,
                "imclef07a": 592,
                "imclef07d": 588,
            },
        }
    )


# ---------------------------------------------------------------------------
# Entry-point group name
# ---------------------------------------------------------------------------

_ENTRY_POINT_GROUP = "hmc_torch.datasets"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Central registry for HMC dataset managers.

    Built-in datasets (arxiv, wos, gofun, aapd, rcv1, eurlex) are
    auto-registered when their modules are imported.  External datasets
    can be registered via:

    1. **Entry points** in ``pyproject.toml`` (scanned automatically).
    2. **Programmatic** ``DatasetRegistry.register(name, factory)``.

    Usage::

        manager = DatasetRegistry.get("wos", data_dir="./data/wos", ...)
        train, valid, test = manager.get_datasets()
    """

    _managers: ClassVar[dict[str, Callable]] = {}
    _defaults: ClassVar[dict[str, dict]] = {}
    _entry_points_scanned: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str | None = None,
        factory: Callable[..., Any] | None = None,
        defaults: dict | None = None,
    ):
        """Register a dataset manager factory.

        Supports three calling conventions:

        1. **Direct call** — pass both *name* and *factory*::

               DatasetRegistry.register("my_data", create_manager,
                                        defaults={"hidden_dim": 256})

        2. **Decorator with arguments** — pass *name* but omit *factory*; returns
           a decorator that will register the decorated class/function::

               @DatasetRegistry.register("my_data", defaults={"hidden_dim": 256})
               class MyManager:
                   ...

        3. **Bare decorator** — pass only *factory*; the dataset name is
           inferred from ``factory.__name__``::

               @DatasetRegistry.register
               def make_my_data(**kw): ...

        Args:
            name: Unique dataset identifier (e.g. ``"my_data"``).  Inferred
                from ``factory.__name__`` when used as a bare decorator.
            factory: Callable that receives ``**kwargs`` and returns a manager
                instance.  When *factory* is omitted (``None``) the method
                acts as a decorator and returns a wrapper that will call
                :meth:`_do_register` later.
            defaults: Optional dictionary of default hyperparameters
                (``hidden_dim``, ``lr``, ``epochs``, ``dropout``, etc.).

        Returns:
            | Direct call: returns *factory* unchanged.
            | Decorator call: returns a decorator function ``(fn) -> fn``
              that registers *fn* and returns it unchanged.

        Raises:
            ValueError: If *name* is already registered.
            TypeError: If the arguments don't match any supported convention.
        """
        # Decorator-with-arguments path: @register("name", defaults=...)
        if factory is None and name is not None and isinstance(name, str):

            def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                cls._do_register(name, fn, defaults)
                return fn

            return _decorator

        # Bare-decorator path: @register
        if factory is not None and name is None:
            cls._do_register(factory.__name__, factory, defaults)
            return factory

        # Direct-call path: register("name", factory, defaults=...)
        if factory is not None and name is not None:
            cls._do_register(name, factory, defaults)
            return factory

        raise TypeError(
            "DatasetRegistry.register() requires either:\n"
            "  (name, factory) — direct call\n"
            "  (name, defaults=...) as decorator — @register('name', ...)\n"
            "  (factory) as bare decorator — @register"
        )

    @classmethod
    def _do_register(
        cls, name: str, factory: Callable[..., Any], defaults: dict | None
    ) -> None:
        """Internal: register without the decorator dispatch logic."""
        if name in cls._managers:
            raise ValueError(
                f"Dataset '{name}' is already registered. "
                f"Use DatasetRegistry.unregister('{name}') first if you "
                f"intend to replace it."
            )
        cls._managers[name] = factory
        if defaults:
            cls._defaults[name] = defaults
        logger.debug("Registered dataset '%s'", name)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a previously registered dataset.

        Does nothing if *name* is not registered.
        """
        cls._managers.pop(name, None)
        cls._defaults.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Instantiate a dataset manager by name.

        This is the primary way to obtain a ready-to-use dataset manager.
        It calls the registered factory with *kwargs* and returns the
        resulting manager instance.

        Args:
            name: Dataset identifier (e.g. ``"wos"``, ``"my_data"``).
            **kwargs: Forwarded to the manager factory.  Common arguments:

                - ``device`` — torch device (``"cpu"``, ``"cuda"``).
                - ``dataset_path`` — root directory for dataset files.
                - ``model_name`` — HuggingFace transformer name (text datasets).
                - ``max_records`` — cap on records to load.
                - ``cache_dir`` — directory for feature cache.
                - ``load_features`` — whether to compute transformer embeddings.

        Returns:
            A manager instance.  Every manager exposes at least these
            attributes: ``input_dim``, ``output_dim``, ``levels_size``,
            ``max_depth``, ``a``, ``edge_index``, ``nodes_idx``,
            ``to_eval``, ``hierarchy_map``, and ``get_datasets()``.

        Raises:
            ValueError: If *name* is not registered.  The error message
                includes the list of available datasets.
        """
        cls._ensure_entry_points()
        if name not in cls._managers:
            available = cls.list_available()
            raise ValueError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {available}. "
                f"Use DatasetRegistry.register('{name}', ...) to add a "
                f"custom dataset, or check the dataset name for typos."
            )
        return cls._managers[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return sorted list of all registered dataset names."""
        cls._ensure_entry_points()
        return sorted(cls._managers.keys())

    @classmethod
    def get_defaults(cls, name: str) -> dict:
        """Return default hyperparameters for *name*, or empty dict."""
        cls._ensure_entry_points()
        return cls._defaults.get(name, {}).copy()

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_entry_points(cls) -> None:
        """Scan entry points once for external dataset registrations."""
        if cls._entry_points_scanned:
            return
        cls._entry_points_scanned = True

        try:
            # Python 3.12+ recommended API; fall back to importlib_metadata
            from importlib.metadata import entry_points  # pylint: disable=import-outside-toplevel
        except ImportError:
            return

        try:
            eps = entry_points(group=_ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.12: entry_points() takes no arguments
            try:
                all_eps = entry_points()
                eps = all_eps.get(_ENTRY_POINT_GROUP, [])
            except (AttributeError, TypeError):
                return
        except (ImportError, AttributeError):
            return

        for ep in eps:
            if ep.name in cls._managers:
                logger.debug(
                    "Entry point '%s' shadows existing dataset — skipped",
                    ep.name,
                )
                continue
            try:
                factory = ep.load()
                cls._managers[ep.name] = factory
                logger.info("Loaded dataset '%s' from entry point", ep.name)
            except (ImportError, AttributeError) as exc:
                logger.warning("Failed to load entry point '%s': %s", ep.name, exc)


# ---------------------------------------------------------------------------
# Convenience: pick defaults for a dataset name
# ---------------------------------------------------------------------------


def pick_defaults(
    registry: DatasetRegistry, _hyper: _HyperDefaults, dataset_name: str
) -> dict:
    """Return the best-guess hyperparameter defaults for *dataset_name*.

    Resolution order:
    1. Explicit defaults registered with the dataset name.
    2. Family-based lookup for GoFun suffixes (``_FUN``, ``_GO``, ``_others``).
    3. Named defaults for text datasets (``wos``, ``arxiv``, ``aapd``, …).
    """
    # 1. Explicit per-dataset defaults (from custom registration)
    explicit = registry.get_defaults(dataset_name)
    if explicit:
        return explicit

    # 2. GoFun ARFF suffixes
    for suffix, key in [
        ("_FUN", "gofun_defaults"),
        ("_GO", "gofun_defaults"),
        ("_others", "gofun_defaults"),
    ]:
        if dataset_name.endswith(suffix):
            return getattr(_hyper, key)

    # 3. Named defaults
    named_map = {
        "wos": _hyper.wos_defaults,
        "aapd": _hyper.aapd_defaults,
        "rcv1": _hyper.rcv1_defaults,
        "eurlex": _hyper.eurlex_defaults,
    }
    if dataset_name in named_map:
        return named_map[dataset_name]

    # 4. Fallback: arxiv defaults
    return _hyper.arxiv_defaults


# Singleton instance for lazy access
_hyper_defaults = _HyperDefaults()
