# 应用写入权限策略（第65节第二阶段）

默认 read_only_demo。本阶段只生成待部署包，没有启用正式写入、重启服务或启动正式 Worker。

## 单一策略及四层执行

backend/app/access_policy.py 是模式与路由的唯一来源。浏览器请求、表单及模型不能改变模式。

| 层 | 实现 | 校验 |
|---|---|---|
| 启动管理器 | local_agent.py | full_app 必须有独立暂存包；固定模式、清单、实现源码及四份适配文件的哈希 |
| 前端 HTTP | application-policy.mjs + 暂存 frontend-guard | 同一份清单；固定后端地址；写请求携带后端令牌，真实性仍由后端认证 |
| 后端 HTTP | ApplicationAccessMiddleware + 暂存 backend guard | 精确方法/UUID路径；保留认证，拒绝客户端身份及模式字段 |
| SQL | write_guard.py + 暂存 SQL adapter | Session 绑定用户、操作和目标；只有经 ORM 对象归属检查的衣物写入可以执行 |

保留现有登录同步的精确例外，不新增用户/身份写权限。Agent 业务预览仍只读，保留既有 Redis 限流。活动私有保护文件及模型网络白名单没有改变。

## 最小 full_app 范围

允许自己的单件衣物新增、读取、PATCH、DELETE、archive、restore、rotate。分类和标签通过原 PATCH API 修改；状态通过归档/恢复API。图片随衣物上传、签名访问及删除，不开放任意文件路径写入/删除。

继续拒绝：批量操作、穿搭保存、家庭/用户资料修改、记录穿戴/清洗、视觉标注、去背景及其他业务写入。

有穿搭、历史、附图等关联记录的衣物删除返回409，避免数据库级联扩大到其他表；进行中图片任务也阻止删除。后续需另行设计并授权关联记录处理。

图片创建失败或数据库事务失败只补偿本次新文件；衣物删除提交后才删除文件。图片路径必须属于认证用户，已有路径不能借 PATCH 重定向。

## 仅图片旋转 Worker

app.workers.authorized_rotation.WardrobeRotationWorkerSettings 只注册 authorized_rotate_item_image_job；没有视觉、去背景或 cron。单并发、一次尝试、禁止自动重试。

旋转 API 在同用户衣物行锁内签发任务ID；入队失败恢复状态。HMAC凭证绑定用户、衣物、方向和任务ID。Worker复核归属及当前状态，复用原旋转实现，通过相同 SQL 保护提交；完成后的凭证重放被拒绝。

本阶段只在独立 Redis 唯一队列实际消费任务，正式 Worker 未启动，旧 Worker 继续停用。

## 暂存、部署条件与回滚

仅生成新目录，不覆盖活动配置：

    backend/.venv/bin/python -B backend/scripts/stage_application_policy.py \
      --mode full_app --output /path/to/new-private-stage-directory

输出 candidate-runtime.json、权限清单及四份适配文件；文件0600、目录0700。未指定模式时仍为只读。

正式启用前仍需单独授权并完成：

1. 为本轮前端代理变更生成和验证独立生产构建；3001当前产物没有更新。
2. 复核源码哈希与候选包；后续源码变化会令旧包失效。
3. 保留原配置、旧构建和进程状态，显式启用可信 full_app 配置，再通过管理器定向加载。
4. 为新旋转 Worker 安全注入相同数据库、图片路径及批准的Redis/队列配置，保留网络、资源保护。不能直接启动旧Worker容器。
5. 检查队列，避免把旧未签发任务交给新消费者。本次不消费历史队列。

回滚使用原配置和旧构建，仅由管理器定向恢复应用；不得清理数据库或启动整套 Compose。本阶段正式环境未变，无需执行回滚。

## 本轮验证

新增后端38项通过：36项权限/适配包测试、2项真实隔离集成。最终受代理错误码改动影响的3项适配测试再次通过。

隔离连接：PostgreSQL 127.0.0.1:15432/wardrowbe_browser_test、Redis 127.0.0.1:16379/1、新临时图片目录。没有迁移；只创建并清理本轮唯一UUID用户，不改人工验收A/B账号。其他TCP连接禁止。

第一项真实集成含19个检查点：认证CRUD、三种签名图片、跨用户拒绝、伪造字段、只读拒绝、分类/标签/状态、真实数据库失败补偿、部分图片失败补偿、SQL反向检查、Session身份、队列失败恢复、签名与重放、关联删除拒绝等。

第二项真实集成验证 API → 独立Redis唯一队列 → 受限ARQ旋转Worker，1成功、0失败。

前端新增3项权限错误映射测试通过；两个历史测试未重跑。TypeScript及相关文件ESLint通过。Python语法通过；没有安装Ruff，未执行。未构建生产版本或调用真实模型。

测试前须安全注入获准的独立 TEST_DATABASE_URL、TEST_REDIS_URL、对应应用连接配置、新临时 STORAGE_PATH，并在导入应用前关闭项目 .env 自动加载。不得回退到开发库。

使用 --noconftest 避免原全局数据库/Redis清理 fixture；真实测试还要求 WARDROBE_POLICY_ISOLATED_TEST=1 及 WARDROBE_POLICY_TEST_REPORT：

    backend/.venv/bin/python -B -m pytest -q --noconftest -c /dev/null \
      -p no:cacheprovider --maxfail=1 \
      backend/tests/test_wardrobe_access_policy.py \
      backend/tests/test_wardrobe_write_isolation.py

测试令牌仅属于新建隔离用户，不是正式会话；不打印令牌、密码或签名URL。
