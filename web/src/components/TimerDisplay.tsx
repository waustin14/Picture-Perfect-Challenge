interface Props {
  remainingSeconds: number | null;
}

export function TimerDisplay({ remainingSeconds }: Props) {
  if (remainingSeconds === null) {
    return <div className="timer timer--waiting">Waiting</div>;
  }

  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  const timeString = minutes > 0
    ? `${minutes}:${seconds.toString().padStart(2, '0')}`
    : `${seconds}s`;

  let timerClass = "timer timer--running";
  if (remainingSeconds <= 0) {
    timerClass = "timer timer--ended";
  } else if (remainingSeconds <= 10) {
    timerClass = "timer timer--warning";
  }

  return <div className={timerClass}>{remainingSeconds <= 0 ? "Time!" : timeString}</div>;
}
