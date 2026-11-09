# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. Email the security report to: [security contact - to be configured]
3. Or use GitHub's private vulnerability reporting feature

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| Critical | Remote code execution, data breach | 24-48 hours |
| High | Authentication bypass, privilege escalation | 7 days |
| Medium | Information disclosure, XSS | 14 days |
| Low | Minor issues | 30 days |

## Security Best Practices

When self-hosting Wardrowbe, follow these recommendations:

### Secrets Management

- Generate strong secrets: `openssl rand -hex 32`
- Never commit `.env` files
- Rotate secrets periodically
- Use different secrets for each environment

### Network Security

- Always use HTTPS in production
- Place behind a reverse proxy (nginx, Caddy, Traefik)
- Use firewall rules to limit exposure
- Consider VPN for admin access

### Authentication

- Use OIDC provider for multi-user deployments
- Enable MFA on your OIDC provider
- Regularly review user access

### Updates

- Keep Docker images updated
- Monitor Dependabot alerts
- Subscribe to security advisories

### Backups

- Regular database backups
- Store backups securely (encrypted)
- Test backup restoration

## Known Security Considerations

### Image Storage

- Uploaded images are stored on disk
- Access controlled via user authentication
- Consider disk encryption for sensitive deployments

### AI Service

- AI requests may contain clothing images
- Use local AI (Ollama) for maximum privacy
- Review AI provider privacy policies if using cloud services

### Database

- Passwords are not stored (OIDC-based auth)
- Session tokens use secure JWT
- Database should be on private network

## Acknowledgments

We appreciate security researchers who help keep Wardrowbe secure. Contributors will be acknowledged (with permission) in release notes.
