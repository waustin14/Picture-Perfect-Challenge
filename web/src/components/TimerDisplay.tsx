interface Props {
  remainingSeconds: number | null;
}

export function TimerDisplay({ remainingSeconds }: Props) {
  if (remainingSeconds === null) {
    return <div className="badge bg-secondary fs-5">Waiting…</div>;
  }
  return <div className="badge bg-info text-dark fs-4">{remainingSeconds}s</div>;
}
