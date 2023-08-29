import asyncio
import logging
import threading
from contextlib import contextmanager

import aiohttp

logger = logging.getLogger(__name__)


def thread_with_aiohttp(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    """Background thread that runs the asyncio event loop.

    Tasks are added to the queue as functions, and run in the event loop. The
    first positional argument to the function is an aiohttp session."""
    asyncio.set_event_loop(loop)

    # Can't use context manager because we are not async
    session = aiohttp.ClientSession()

    async def task_runner():
        while True:
            # Note: This await is what pumps the event loop and lets tasks run
            async_func, args, kwargs = await queue.get()
            if async_func is None:
                # This is a signal to stop processing events
                break
            asyncio.create_task(async_func(session, *args, **kwargs))

    loop.run_until_complete(task_runner())
    # The thread is done, so wait for all tasks created above to be completed
    # (need outer loop because tasks could schedule other tasks)
    while tasks := asyncio.all_tasks(loop):
        for task in tasks:
            loop.run_until_complete(task)

    # Need to wait for a close since there might be network requests in flight
    loop.run_until_complete(session.close())


_send_events_thread: threading.Thread | None = None
_send_event_queue: asyncio.Queue | None = None
_send_event_loop: asyncio.AbstractEventLoop | None = None

_monitor_thread: threading.Thread | None = None


def run_monitor_thread(loop, queue):
    # block until main thread exits
    threading.main_thread().join()
    # tell the background event loop to exit
    asyncio.run_coroutine_threadsafe(queue.put((None, None, None)), loop)


@contextmanager
def ensure_background_thread():
    """Ensure that the background event-sending thread is running.

    Usage::
        async def send_event(session, arg1, arg2):
            # do something
            pass

        with ensure_background_thread() as call_async_function:
            call_async_function(send_event, *args, **kwargs)
    """
    global _send_events_thread
    global _send_event_queue
    global _send_event_loop
    global _monitor_thread
    if _send_events_thread is None:
        _send_event_queue = asyncio.Queue()
        _send_event_loop = asyncio.new_event_loop()
        _send_events_thread = threading.Thread(
            target=thread_with_aiohttp, args=(_send_event_loop, _send_event_queue)
        )
        _monitor_thread = threading.Thread(
            target=run_monitor_thread, args=(_send_event_loop, _send_event_queue)
        )
        _monitor_thread.daemon = True
        _monitor_thread.start()
        _send_events_thread.start()
        logger.debug("Started send events thread")

    def call_async_function(func, *args, **kwargs):
        if not _send_event_queue or not _send_event_loop:
            # this should never happen
            raise Exception("Event queue not initialized")
        asyncio.run_coroutine_threadsafe(
            _send_event_queue.put((func, args, kwargs)), _send_event_loop
        )

    yield call_async_function
