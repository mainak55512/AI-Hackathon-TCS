import { get } from "svelte/store";
import { authToken } from "./auth";

export function getToken() {
    let token = get(authToken);
    return token;
}
