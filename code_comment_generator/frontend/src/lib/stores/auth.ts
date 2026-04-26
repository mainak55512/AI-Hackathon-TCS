import { writable } from "svelte/store";

const isBrowser = typeof window !== "undefined";

export const authToken = writable(
    isBrowser ? localStorage.getItem("jwt") : null,
);

authToken.subscribe((token) => {
    if (!isBrowser) return;
    if (token) localStorage.setItem("jwt", token);
    else localStorage.removeItem("jwt");
});
