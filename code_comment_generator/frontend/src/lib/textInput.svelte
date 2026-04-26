<script>
  // import {getToken} from './stores/fetchToken'
  import { authToken } from "./stores/auth";
  // let token = getToken()

  import { onMount } from "svelte";

  let snippet = ""
  let llm_output = ""

  onMount(() => {
    console.log("Token on mount:", $authToken);
  });
  async function generate_cmt() {
    if (snippet) {
      const res = await fetch('/api/comments', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${$authToken}`,
        },
        body: JSON.stringify({ snippet })
      })

      const data = await res.json()
      llm_output = data.comments
    }
  }
</script>
<div> 
  Code Snippet:
  <textarea bind:value={snippet}></textarea>
  <button on:click={() => generate_cmt()}>Generate Comments</button>
    <br>
    <pre>{llm_output}</pre>
</div>
