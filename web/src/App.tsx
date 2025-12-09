import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { JoinPage } from "./pages/JoinPage";
import { PlayerGamePage } from "./pages/PlayerGamePage";
import { AdminCreateGamePage } from "./pages/AdminCreateGamePage";
import { AdminGamePage } from "./pages/AdminGamePage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/join" />} />
        <Route path="/join" element={<JoinPage />} />
        <Route path="/game/:gameId" element={<PlayerGamePage />} />
        <Route path="/admin" element={<AdminCreateGamePage />} />
        <Route path="/admin/game/:gameId" element={<AdminGamePage />} />
      </Routes>
    </BrowserRouter>
  );
}
