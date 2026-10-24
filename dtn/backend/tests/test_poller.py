import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from app.db.models import Device, DeviceStatus
from app.services.events import EventManager
from app.services.poller import DevicePoller


def test_poller_persists_validated_real_status(settings) -> None:
    from app.db.models import Base
    from app.db.session import create_db_engine, create_session_factory

    engine = create_db_engine("sqlite://")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    poller = DevicePoller(settings, sessions, EventManager())
    poller._ensure_registry()
    config = settings.nodes[0]
    from app.schemas.devices import StatusPayload

    payload = StatusPayload(
        node_id="master",
        role="master",
        hostname="dfl-master",
        sampled_at=datetime.now(timezone.utc),
        cpu_utilization_percent=20.0,
        gpu_utilization_percent=0.0,
        memory={"total_bytes": 100, "available_bytes": 60, "used_bytes": 40},
        temperatures_celsius={"cpu-thermal": 45.0},
    )
    poller._store_status(config, payload, payload.model_dump(mode="json"), payload.sampled_at)
    with sessions() as session:
        device = session.get(Device, "master")
        history = session.query(DeviceStatus).filter_by(node_id="master").one()
        assert device is not None and device.online is True
        assert history.memory["available_bytes"] == 60
    engine.dispose()


def test_poller_marks_failed_device_offline(settings) -> None:
    from app.db.models import Base
    from app.db.session import create_db_engine, create_session_factory

    engine = create_db_engine("sqlite://")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    poller = DevicePoller(settings, sessions, EventManager())
    poller._ensure_registry()
    config = settings.nodes[0]
    with sessions() as session:
        device = session.get(Device, config.node_id)
        assert device is not None
        device.online = True
        session.commit()

    was_online, _, last_seen = poller._mark_offline(config, "connection timed out")
    assert was_online is True
    assert last_seen is None
    with sessions() as session:
        device = session.get(Device, config.node_id)
        assert device is not None
        assert device.online is False
        assert device.last_error == "connection timed out"
    engine.dispose()


def test_poller_keeps_running_until_clean_shutdown(settings) -> None:
    from app.db.models import Base
    from app.db.session import create_db_engine, create_session_factory

    engine = create_db_engine("sqlite://")
    Base.metadata.create_all(engine)
    poller = DevicePoller(
        replace(settings, poll_interval_seconds=1),
        create_session_factory(engine),
        EventManager(),
    )
    calls = 0

    async def count_poll() -> None:
        nonlocal calls
        calls += 1

    poller.poll_once = count_poll

    async def exercise() -> None:
        await poller.start()
        await asyncio.sleep(1.1)
        await poller.stop()

    asyncio.run(exercise())
    assert calls >= 2
    engine.dispose()
