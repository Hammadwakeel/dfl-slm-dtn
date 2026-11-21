"""Authenticated catalog for the supported official image datasets."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.security import require_api_token
from app.services.datasets import (
    DatasetError,
    DatasetStore,
)
from app.services.cluster import cluster_nodes
from app.services.dataset_pipeline import ship_global_validation
from app.services.preprocessing import PreprocessingError
from app.services.dataset_samples import get_dataset_samples
from app.services.initial_models import (
    InitialModelError,
    delete_initial_model,
    get_initial_model_report,
)
from app.services.worker_shards import WorkerShardError

router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_api_token)],
)

SUPPORTED_DATASETS = frozenset({"mnist", "cifar10", "fashion_mnist", "femnist"})


def get_dataset_store(request: Request) -> DatasetStore:
    return request.app.state.datasets


def _call(callback: Any) -> Any:
    try:
        return callback()
    except DatasetError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("")
def list_datasets(store: DatasetStore = Depends(get_dataset_store)) -> list[dict[str, Any]]:
    return _call(
        lambda: [
            item
            for item in store.list_metadata()
            if item.get("dataset_name") in SUPPORTED_DATASETS
        ]
    )


def _require_supported_dataset(dataset_name: str) -> None:
    if dataset_name not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Only the official MNIST and CIFAR-10 datasets are enabled",
        )


@router.get("/{dataset_name}/preprocessing-report")
def get_latest_preprocessing_report(
    dataset_name: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> dict[str, Any]:
    _require_supported_dataset(dataset_name)
    return _call(lambda: store.get_latest_preprocessing_report(dataset_name))


@router.get("/{dataset_name}/samples")
def get_samples(
    dataset_name: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> list[dict[str, Any]]:
    _require_supported_dataset(dataset_name)
    return _call(lambda: get_dataset_samples(store, dataset_name))


@router.get("/{dataset_name}/initial-model")
def get_dataset_initial_model(
    dataset_name: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> dict[str, Any]:
    _require_supported_dataset(dataset_name)
    try:
        report = get_initial_model_report(store, dataset_name)
    except DatasetError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except InitialModelError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if report is None:
        raise HTTPException(status_code=404, detail=f"Initial model not found for {dataset_name}")
    return report


@router.delete("/{dataset_name}/initial-model", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_initial_model(
    dataset_name: str,
    request: Request,
    store: DatasetStore = Depends(get_dataset_store),
) -> Response:
    _require_supported_dataset(dataset_name)
    try:
        delete_initial_model(
            store,
            cluster_nodes(request.app.state.settings, request.app.state.session_factory),
            dataset_name,
        )
    except DatasetError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except (InitialModelError, WorkerShardError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{dataset_name}/global-validation")
def install_dataset_global_validation(
    dataset_name: str,
    request: Request,
    store: DatasetStore = Depends(get_dataset_store),
) -> dict[str, Any]:
    """Ship the shared class-balanced validation subset of the installed shards to every worker."""
    _require_supported_dataset(dataset_name)
    try:
        return ship_global_validation(
            store,
            cluster_nodes(request.app.state.settings, request.app.state.session_factory),
            dataset_name,
        )
    except DatasetError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except PreprocessingError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except WorkerShardError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/{dataset_name}")
def get_dataset(
    dataset_name: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> dict[str, Any]:
    _require_supported_dataset(dataset_name)
    return _call(lambda: store.get_metadata(dataset_name))
