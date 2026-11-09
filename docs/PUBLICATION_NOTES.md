# 发布说明

本仓库在 [Wardrowbe](https://github.com/Anyesh/wardrowbe) 基础上扩展，保留原 MIT 许可和版权说明。

- 服务器主机、SSH 用户、远程根目录、GPU UUID 和本机用户路径使用示例值。
- 示例 GPU UUID 为 GPU-11111111-1111-1111-1111-111111111111；示例主机 inference.example.invalid 无法用于真实连接。
- 实际部署需准备自己的可信配置和私有启动保护包，并重新核验源码摘要与权限。校验不通过时停止。
- .env、凭据、运行日志、真实图片、数据库、模型权重和私有运行包均不包含在本源码快照中。
- 仅保留手动触发的 Publication checks，执行静态检查。该工作流不自动部署或推理。
- 使用 python3 scripts/check_publication.py 执行标准库静态检查。

技术验证结果见 [功能与验证范围](FINAL_PROJECT_REPORT.md)，运行步骤见 [部署指南](DEPLOYMENT.md)。
