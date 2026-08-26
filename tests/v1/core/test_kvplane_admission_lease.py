from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.core import kv_cache_manager
from vllm.v1.request import Request


def _request(
    *,
    request_id: str = "cmpl-request-a",
    external_req_id: str | None = None,
    **overrides: object,
):
    extra_args: dict[str, object] = {
        "kvplane_admit_prefix_cache": "admit",
        "kvplane_admission_lease_id": "lease-a",
        "kvplane_admission_request_id": external_req_id or request_id,
        "kvplane_admission_epoch": 7,
        "kvplane_admission_issued_at_ns": 1_000,
        "kvplane_admission_expires_at_ns": 2_000,
    }
    extra_args.update(overrides)
    return SimpleNamespace(
        request_id=request_id,
        external_req_id=external_req_id,
        num_tokens=32,
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


def _write_epoch(path, epoch: object, *, schema: str = "kvplane-pressure-epoch.v1"):
    path.write_text(
        f'{{"schema":"{schema}","pressure_epoch":{epoch}}}',
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _reset_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", raising=False)
    monkeypatch.setattr(kv_cache_manager.time, "monotonic_ns", lambda: 1_100)
    kv_cache_manager.install_kvplane_pressure_epoch_provider(None)
    yield
    kv_cache_manager.install_kvplane_pressure_epoch_provider(None)


def test_file_epoch_provider_allows_monotonic_incremental_publication(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))
    request = _request()

    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )
    kv_cache_manager._kvplane_record_prefix_cache_publication(request, 16)
    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )
    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=32
    )


def test_file_epoch_rollover_revokes_old_lease(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 8)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))

    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        _request(), publication_sequence=16
    )


@pytest.mark.parametrize(
    ("contents", "exists"),
    [
        ('{"schema":"wrong","pressure_epoch":7}', True),
        ('{"schema":"kvplane-pressure-epoch.v1","pressure_epoch":true}', True),
        ("not-json", True),
        ("", False),
    ],
)
def test_invalid_or_missing_epoch_file_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, contents: str, exists: bool
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    if exists:
        epoch_file.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))

    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        _request(), publication_sequence=16
    )


def test_request_identity_mismatch_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))

    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        _request(kvplane_admission_request_id="cmpl-other"),
        publication_sequence=16,
    )


def test_external_request_identity_allows_randomized_internal_id(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))

    request = _request(
        request_id="cmpl-request-a-deadbeef",
        external_req_id="cmpl-request-a",
    )
    assert request.request_id != request.external_req_id
    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )
    kv_cache_manager._kvplane_record_prefix_cache_publication(request, 16)
    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )


def test_external_request_identity_mismatch_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))

    request = _request(
        request_id="cmpl-request-a-deadbeef",
        external_req_id="cmpl-other",
        kvplane_admission_request_id="cmpl-request-a",
    )
    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )


def test_engine_core_request_preserves_external_identity() -> None:
    params = SamplingParams(max_tokens=1)
    engine_request = EngineCoreRequest(
        request_id="cmpl-request-a-deadbeef",
        external_req_id="cmpl-request-a",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.request_id == "cmpl-request-a-deadbeef"
    assert request.external_req_id == "cmpl-request-a"


def test_native_failure_does_not_consume_publication_sequence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))
    request = _request()

    class FailingCoordinator:
        def cache_blocks(self, request, num_computed_tokens):
            raise RuntimeError("native publication failed")

    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.enable_caching = True
    manager.coordinator = FailingCoordinator()

    with pytest.raises(RuntimeError, match="native publication failed"):
        manager.cache_blocks(request, 16)
    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request, publication_sequence=16
    )


def test_commit_handshake_defers_until_post_execute(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))
    request = _request(kvplane_admission_commit_handshake=True)

    class Coordinator:
        def __init__(self) -> None:
            self.publications: list[int] = []

        def cache_blocks(self, request, num_computed_tokens):
            self.publications.append(num_computed_tokens)
            return 1

    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.enable_caching = True
    manager.coordinator = Coordinator()
    manager.log_stats = False

    manager.cache_blocks(request, 16)
    assert manager.coordinator.publications == []

    manager.commit_kvplane_cache_blocks(request, 16)
    assert manager.coordinator.publications == [16]


def test_commit_handshake_revalidates_epoch_after_execution(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))
    request = _request(kvplane_admission_commit_handshake="enabled")

    class Coordinator:
        def __init__(self) -> None:
            self.publications: list[int] = []

        def cache_blocks(self, request, num_computed_tokens):
            self.publications.append(num_computed_tokens)
            return 1

    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.enable_caching = True
    manager.coordinator = Coordinator()
    manager.log_stats = False

    manager.cache_blocks(request, 16)
    _write_epoch(epoch_file, 8)
    manager.commit_kvplane_cache_blocks(request, 16)

    assert manager.coordinator.publications == []


def test_publication_sequence_is_scoped_by_request_and_epoch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch_file = tmp_path / "pressure-epoch.json"
    _write_epoch(epoch_file, 7)
    monkeypatch.setenv("VLLM_KVPLANE_PRESSURE_EPOCH_FILE", str(epoch_file))
    request_a = _request(request_id="cmpl-a")
    request_b = _request(request_id="cmpl-b")

    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request_a, publication_sequence=16
    )
    kv_cache_manager._kvplane_record_prefix_cache_publication(request_a, 16)

    assert not kv_cache_manager._kvplane_allows_prefix_cache_write(
        request_a, publication_sequence=16
    )
    assert kv_cache_manager._kvplane_allows_prefix_cache_write(
        request_b, publication_sequence=16
    )


def test_legacy_boolean_path_is_unchanged() -> None:
    request = SimpleNamespace(
        request_id="legacy",
        sampling_params=SimpleNamespace(
            extra_args={"kvplane_admit_prefix_cache": "admit"}
        ),
    )

    assert kv_cache_manager._kvplane_allows_prefix_cache_write(request)
