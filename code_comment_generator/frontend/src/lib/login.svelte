<script>
  import {authToken} from './stores/auth'
  import TextInput from './textInput.svelte';
  let username = ''
  let password = ''
  let error = ''
  let loading = false

  let isLoggedIn = false
 
  async function handleLogin() {
    if (!username || !password) {
      error = 'Please fill in all fields.'
      return
    }
 
    loading = true
    error = ''
 
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      })
 
      const data = await res.json()
      if (res.ok) {
        authToken.set(data.token)
        isLoggedIn = true
      } else {
        error = data.msg || 'Invalid credentials.'
      }
    } catch (err) {
      error = 'Could not connect to server.'
    } finally {
      loading = false
    }
  }
</script>
 
{#if !isLoggedIn}
<div class="page">
  <div class="card">
    <div class="brand">
      <span class="dot"></span>
      <h1>Welcome back</h1>
    </div>
    <p class="subtitle">Sign in to continue</p>
 
    <div class="form">
      <div class="field">
        <label for="username">Username</label>
        <input
          id="username"
          type="text"
          placeholder="Enter username"
          bind:value={username}
          on:keydown={(e) => e.key === 'Enter' && handleLogin()}
        />
      </div>
 
      <div class="field">
        <label for="password">Password</label>
        <input
          id="password"
          type="password"
          placeholder="Enter password"
          bind:value={password}
          on:keydown={(e) => e.key === 'Enter' && handleLogin()}
        />
      </div>
 
      {#if error}
        <p class="error">{error}</p>
      {/if}
 
      <button on:click={handleLogin} disabled={loading}>
        {loading ? 'Signing in...' : 'Sign In'}
      </button>
    </div>
  </div>
</div>
{/if}
{#if isLoggedIn}
  <TextInput></TextInput>
{/if}
