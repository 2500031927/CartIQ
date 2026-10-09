import { useCallback, useRef, useState } from "react";

/** Wraps an API call in an idle / loading / success / error state machine. */
export default function usePrediction(apiFn) {
  const [state, setState] = useState({ status: "idle", data: null, error: null });
  const lastPayload = useRef(null);

  const run = useCallback(
    async (payload) => {
      lastPayload.current = payload;
      setState({ status: "loading", data: null, error: null });
      try {
        const data = await apiFn(payload);
        setState({ status: "success", data, error: null });
      } catch (error) {
        setState({ status: "error", data: null, error });
      }
    },
    [apiFn]
  );

  const retry = useCallback(() => {
    if (lastPayload.current) run(lastPayload.current);
  }, [run]);

  return { state, run, retry };
}
