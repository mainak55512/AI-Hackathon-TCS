import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'
import CyberTerminalDashboard from './pages/CyberTerminalDashboard'

function App() {
  const [count, setCount] = useState(0)

  return (
    <CyberTerminalDashboard />
  )
}

export default App
