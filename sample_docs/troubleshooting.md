# Troubleshooting

## I did not receive the verification email

Check your spam folder first. Verification emails are sent from
no-reply@acme.example. If it still has not arrived after 15 minutes, request a
new one from the login screen. Some corporate mail filters delay external mail.

## "Invalid API key" errors

This error means the key is wrong, has been revoked, or belongs to a different
workspace. Generate a new key from **Settings → API keys** and update your
application. Remember that keys are scoped to a single workspace.

## A project is stuck in "provisioning"

Provisioning usually completes within two minutes. If a project is stuck for
longer than ten minutes, it has likely failed. Delete the project and create a
new one in the same region. If it fails again, contact support with the project
ID.

## Rate limit reached

The API returns HTTP 429 when you exceed your plan's request rate. Wait for the
window to reset, then retry with exponential backoff. Team and Enterprise plans
have higher limits.
