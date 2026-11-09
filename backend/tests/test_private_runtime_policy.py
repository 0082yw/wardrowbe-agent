"""模式策略及实际私有保护的离线测试；不连接数据库、SSH 或模型服务。"""

import ast
import contextvars
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import httpx
import pytest

from app.services.agent_model_config import AgentModelConfig
from app.services.agent_runtime_policy import ModelRuntimePolicy
from app.services.local_agent_model import LocalAgentTextModel
from app.services.ollama_agent_model import OllamaModelError
from app.services.remote_agent_model import RemoteAgentTextModel


@pytest.fixture(params=['local_cpu','remote_gpu'])
def policy(request):
    return ModelRuntimePolicy(AgentModelConfig(request.param,
        'qwen3:0.6b-cpu' if request.param=='local_cpu' else 'qwen3:4b'))


@pytest.mark.parametrize('port',[5432,6379,11434,11435])
def test_only_dependencies_and_selected_model_port_are_allowed(policy,port):
    args=(None,('127.0.0.1',port))
    if port in (5432,6379,policy.model_port):policy.network_guard('socket.connect',args)
    else:
        with pytest.raises(PermissionError):policy.network_guard('socket.connect',args)


@pytest.mark.parametrize('address',[('inference.example.invalid',11434),('8.8.8.8',443),('127.0.0.1',8000),
    ('127.0.0.1',22),('localhost',11435),('::1',11435),'/tmp/arbitrary.sock'])
def test_external_unknown_ports_and_unix_sockets_are_rejected(policy,address):
    with pytest.raises(PermissionError):policy.network_guard('socket.connect',(None,address))


def test_dns_and_bind_protections_remain_restricted(policy):
    policy.network_guard('socket.getaddrinfo',('localhost',))
    policy.network_guard('socket.bind',(None,('127.0.0.1',8001)))
    for event,args in [('socket.getaddrinfo',('external.invalid',)),
                       ('socket.bind',(None,('0.0.0.0',8001))),('socket.bind',(None,('127.0.0.1',3001)))]:
        with pytest.raises(PermissionError):policy.network_guard(event,args)


def model_request(policy,**changes):
    data={'model':policy.config.model_name,'tools':[],'reasoning_effort':'none','max_tokens':256,
          'stream':False,'messages':[{'role':'user','content':'offline'}],**changes}
    return httpx.Request('POST',f'http://127.0.0.1:{policy.model_port}/v1/chat/completions',json=data)


def test_attestation_never_substitutes_cpu_for_remote(policy):
    model=(LocalAgentTextModel() if policy.config.execution_mode=='local_cpu'
           else RemoteAgentTextModel(model=policy.config.model_name))
    policy.attest_model(model)
    other=(RemoteAgentTextModel(model='qwen3:4b') if policy.config.execution_mode=='local_cpu'
           else LocalAgentTextModel())
    with pytest.raises(AssertionError):policy.attest_model(other)


def test_http_model_and_metadata_operations_are_mode_scoped(policy):
    assert policy.validate_http(model_request(policy))
    path='/api/ps' if policy.config.execution_mode=='local_cpu' else '/api/tags'
    assert not policy.validate_http(httpx.Request('GET',f'http://127.0.0.1:{policy.model_port}{path}'))
    for path in ['/api/pull','/api/create','/api/generate','/api/chat']:
        with pytest.raises(PermissionError):
            policy.validate_http(httpx.Request('POST',f'http://127.0.0.1:{policy.model_port}{path}'))


@pytest.mark.parametrize('changes',[{'model':'another-model'},{'tools':[{}]},
    {'reasoning_effort':'high'},{'max_tokens':4096},{'stream':True}])
def test_unapproved_generation_parameters_are_rejected(policy,changes):
    with pytest.raises(AssertionError):policy.validate_http(model_request(policy,**changes))


@pytest.mark.parametrize('header',['Authorization','Cookie'])
def test_authentication_material_never_reaches_model(policy,header):
    request=model_request(policy);request.headers[header]='isolated-private-marker'
    with pytest.raises(PermissionError) as error:policy.validate_http(request)
    assert 'isolated-private-marker' not in str(error.value)


@pytest.fixture
def private_sources():
    private=Path.home()/'.local/state/wardrowbe-agent/local-runtime'
    if not (private/'backend.py').is_file():pytest.skip('此机器没有私有运行包；纯策略测试仍运行')
    return private,{n:(private/n).read_text() for n in ('backend.py','backend-entry.py')}


def extract(source,name,namespace):
    node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<offline-private-extract>','exec'),namespace)
    return namespace[name]


def test_actual_private_network_function_uses_selected_policy(private_sources,policy):
    _,sources=private_sources
    fn=extract(sources['backend.py'],'network_guard',{'POLICY':policy})
    fn('socket.connect',(None,('127.0.0.1',policy.model_port)))
    with pytest.raises(PermissionError):fn('socket.connect',(None,('127.0.0.1',11435 if policy.model_port==11434 else 11434)))


@pytest.mark.asyncio
async def test_actual_http_hook_keeps_generation_budget_and_no_network(private_sources,policy):
    _,sources=private_sources
    ctx=contextvars.ContextVar('offline-generation',default=None)
    ns={'POLICY':policy,'calls':0,'local':__import__('app.services.local_agent_model',fromlist=['_fail']),
        'wincheck':Mock(return_value=4*2**30),'ctx':ctx}
    fn=extract(sources['backend.py'],'before_http',ns)
    for _ in range(3):await fn(model_request(policy))
    with pytest.raises(OllamaModelError) as error:await fn(model_request(policy))
    assert error.value.code=='session_request_budget' and ns['calls']==3
    assert ns['wincheck'].call_count==4


def test_actual_client_hook_requires_disabled_proxies_and_redirects(private_sources):
    _,sources=private_sources
    old=Mock();ns={'old_client_init':old,'before_http':Mock(),'after_http':Mock()}
    fn=extract(sources['backend.py'],'client_init',ns)
    for kwargs in ({},{'trust_env':True,'follow_redirects':False},{'trust_env':False,'follow_redirects':True}):
        with pytest.raises(PermissionError):fn(object(),**kwargs)
    fn(object(),trust_env=False,follow_redirects=False)
    assert old.call_count==1


def test_private_backups_hash_chain_and_auth_sql_are_preserved(private_sources):
    private,sources=private_sources
    backups=sorted((private/'backups').glob('lesson53-*'))
    assert backups
    backup=backups[-1]
    before=ast.parse((backup/'backend.py').read_text());after=ast.parse(sources['backend.py'])
    sql=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sql_guard')
    assert ast.dump(sql(before))==ast.dump(sql(after))
    auth=lambda source:next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Constant)
        and isinstance(n.value,str) and 'def install_and_run' in n.value)
    assert auth((backup/'backend-entry.py').read_text())==auth(sources['backend-entry.py'])
    config=json.loads((Path.home()/'.config/wardrowbe-agent/local-runtime.json').read_text())
    assert all(hashlib.sha256((private/n).read_bytes()).hexdigest()==v for n,v in config['guard_hashes'].items())
    state=json.loads((private/'auth-state.json').read_text())
    assert state['original_bootstrap_sha256']==hashlib.sha256(sources['backend.py'].encode()).hexdigest()


def test_actual_authentication_and_sql_self_checks_run_without_database(private_sources,monkeypatch,capsys):
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.services.user_service import UserService
    _,sources=private_sources
    auth=next(n.value for n in ast.walk(ast.parse(sources['backend-entry.py'])) if isinstance(n,ast.Constant)
        and isinstance(n.value,str) and 'def install_and_run' in n.value)
    engine=create_async_engine('postgresql+asyncpg://wardrowbe_agent_dev:offline@127.0.0.1:1/wardrowbe_agent_dev')
    ctx=contextvars.ContextVar('offline-sql',default=None)
    ns={'ctx':ctx,'engine':engine}
    sql=extract(sources['backend.py'],'sql_guard',ns)
    event.listen(engine.sync_engine,'before_cursor_execute',sql)
    dispatch=Mock()
    scope={'engine':engine,'ctx':ctx,'sql_guard':sql,'emit':Mock(),'observe_request':dispatch}
    policy={'user_id':str(UUID(int=10)),'identity_id':str(UUID(int=11)),
        'email':'offline@example.invalid','display_name':'Offline Test','subject':'offline-test','avatar_url':None}
    namespace={'scope':scope,'policy':policy,'original_run':Mock(),'contextvars':contextvars,
        'datetime':datetime,'UTC':UTC,'UUID':UUID,'json':json,'sys':sys,'time':__import__('time'),
        'SimpleNamespace':SimpleNamespace}
    monkeypatch.setattr(sys,'argv',['offline-private-check','--auth-check-only'])
    monkeypatch.setattr(UserService,'sync_from_oidc',UserService.sync_from_oidc)
    install=extract(auth,'install_and_run',namespace)
    app=SimpleNamespace(user_middleware=[SimpleNamespace(kwargs={'dispatch':dispatch})],middleware_stack=None)
    install(app)
    value=json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert value['authentication_guard_tests_passed']==31
    assert value['database_mutations_executed']==value['model_requests']==0
    namespace['original_run'].assert_not_called()


def test_remote_launcher_uses_remote_check_and_never_starts_local_ollama(tmp_path,monkeypatch):
    script=Path(__file__).parents[1]/'scripts/local_agent.py'
    spec=importlib.util.spec_from_file_location('private_mode_runtime',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'PRIVATE',tmp_path)
    build=tmp_path/'build';build.mkdir();(build/'BUILD_ID').write_text('offline-build')
    manager=module.Manager({'legacy_heartbeat':'offline','build':str(build),'guard_hashes':{},'node':'offline-node',
                            'frontend_internal_url':'http://127.0.0.1:3001'})
    monkeypatch.setenv('WARDROBE_AGENT_MODEL_MODE','remote_gpu')
    monkeypatch.setenv('WARDROBE_AGENT_REMOTE_MODEL','qwen3:4b')
    monkeypatch.setattr(module,'docker_dependencies',Mock())
    monkeypatch.setattr(module,'port_open',lambda port:False)
    for name in ['ensure_monitor','baseline','preflight_backend','save']:
        monkeypatch.setattr(manager,name,Mock())
    monkeypatch.setattr(manager,'current_resources',Mock(return_value={'windows_gib':1.5}))
    cpu=Mock();monkeypatch.setattr(manager,'ensure_ollama',cpu)
    readiness=AsyncMock();monkeypatch.setattr(RemoteAgentTextModel,'check_ready',readiness)
    monkeypatch.setattr(module,'wait_ready',Mock())
    monkeypatch.setattr(module,'request',Mock())
    monkeypatch.setattr(__import__('dotenv'),'dotenv_values',lambda path:{'NEXTAUTH_SECRET':'offline-only'})
    spawn=Mock(return_value={'pid':1});monkeypatch.setattr(manager,'spawn',spawn)
    manager.start()
    readiness.assert_awaited_once();cpu.assert_not_called()
    assert [c.args[0] for c in spawn.call_args_list]==['backend','frontend']


def test_running_cpu_process_is_not_silently_relabelled_remote(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('idempotent_mode_runtime',Path(__file__).parents[1]/'scripts/local_agent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'PRIVATE',tmp_path)
    manager=module.Manager({'legacy_heartbeat':'offline','guard_hashes':{}})
    manager.state['processes']={n:{'record':{'pid':i},'owned':True} for i,n in enumerate(['backend','frontend'],10)}
    monkeypatch.setattr(module,'docker_dependencies',Mock())
    monkeypatch.setattr(module,'port_open',lambda port:True)
    monkeypatch.setattr(module,'same_process',lambda record:True)
    monkeypatch.setenv('WARDROBE_AGENT_MODEL_MODE','remote_gpu')
    monkeypatch.setenv('WARDROBE_AGENT_REMOTE_MODEL','qwen3:4b')
    stop=Mock();monkeypatch.setattr(manager,'stop_one',stop)
    with pytest.raises(module.RuntimeErrorSafe,match='不会静默切换'):manager.start()
    stop.assert_not_called()


@pytest.mark.parametrize('mode',['local_cpu','remote_gpu'])
def test_actual_entry_attests_audited_route_and_selected_dependency_without_binding(private_sources,monkeypatch,capsys,mode):
    from fastapi import FastAPI
    from app.api import agent
    from app import config,database
    from app.services.user_service import UserService
    _,sources=private_sources
    app=FastAPI();app.include_router(agent.router,prefix='/api/v1')
    async def restricted_sync(*args,**kwargs):raise AssertionError('offline: no login')
    monkeypatch.setattr(UserService,'sync_from_oidc',restricted_sync)
    monkeypatch.setattr(database,'engine',SimpleNamespace(url=SimpleNamespace(host='127.0.0.1',port=5432,
        database='wardrowbe_agent_dev',username='wardrowbe_agent_dev')))
    monkeypatch.setattr(config,'get_settings',lambda:SimpleNamespace(
        storage_path='/home/developer/.local/share/wardrowbe-agent/images',redis_url='redis://127.0.0.1:6379/15',
        dev_mode=True,ai_internal_enabled=False,ai_text_enabled=False,ai_vision_enabled=False))
    monkeypatch.setenv('WARDROBE_AGENT_MODEL_MODE',mode)
    monkeypatch.setenv('WARDROBE_AGENT_REMOTE_MODEL','qwen3:4b')
    monkeypatch.setattr(sys,'argv',['offline-entry','--lesson48-no-bind'])
    saved=Mock();fn=extract(sources['backend-entry.py'],'_attest_loaded',{'_saved_run':saved,'json':json,'sys':sys})
    fn(app)
    rows=[json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert rows[0]['audited_route_registered'] and rows[0]['audited_route_methods']==['POST']
    assert rows[-1]['execution_mode']==mode
    assert rows[-1]['real_remote_model_injected']==(mode=='remote_gpu')
    assert rows[-1]['gpu_verified'] is not True
    saved.assert_not_called()


@pytest.mark.parametrize("port", [11434, 11435, 8001])
def test_remote_show_is_readonly_and_only_on_fixed_tunnel_port(port):
    p = ModelRuntimePolicy(AgentModelConfig("remote_gpu", "qwen3:4b-instruct-2507-q4_K_M"))
    request = httpx.Request("POST", f"http://127.0.0.1:{port}/api/show", json={"model": p.config.model_name})
    if port == 11435: assert not p.validate_http(request)
    else:
        with pytest.raises(PermissionError): p.validate_http(request)


@pytest.mark.parametrize("body", [
    {"model": "another"}, {"model": "qwen3:4b-instruct-2507-q4_K_M", "user_id": "untrusted"},
    {"model": "qwen3:4b-instruct-2507-q4_K_M", "keep_alive": -1}, {},
])
def test_remote_show_rejects_extra_parameters_and_other_models(body):
    p = ModelRuntimePolicy(AgentModelConfig("remote_gpu", "qwen3:4b-instruct-2507-q4_K_M"))
    with pytest.raises(AssertionError):
        p.validate_http(httpx.Request("POST", "http://127.0.0.1:11435/api/show", json=body))


@pytest.mark.parametrize("url", [
    "http://inference.example.invalid:11435/api/show", "http://127.0.0.1:11435/api/pull",
    "http://127.0.0.1:11435/api/show?other=1", "https://127.0.0.1:11435/api/show",
])
def test_new_readonly_exception_never_expands_to_other_destinations(url):
    p = ModelRuntimePolicy(AgentModelConfig("remote_gpu", "qwen3:4b-instruct-2507-q4_K_M"))
    with pytest.raises(PermissionError):
        p.validate_http(httpx.Request("POST", url, json={"model": p.config.model_name}))


def test_remote_instruct_body_may_omit_switch_but_cpu_still_requires_none():
    for mode in ["local_cpu", "remote_gpu"]:
        p = ModelRuntimePolicy(AgentModelConfig(mode, "qwen3:0.6b-cpu" if mode == "local_cpu"
                                               else "qwen3:4b-instruct-2507-q4_K_M"))
        request = model_request(p)
        body = json.loads(request.content); body.pop("reasoning_effort")
        request = httpx.Request("POST", str(request.url), json=body)
        if mode == "remote_gpu": assert p.validate_http(request)
        else:
            with pytest.raises(KeyError): p.validate_http(request)


@pytest.mark.parametrize("changes", [{"reasoning_effort": None}, {"think": False}, {"options": {"num_predict": 256}}])
def test_remote_generation_rejects_null_or_native_control_fields(changes):
    p = ModelRuntimePolicy(AgentModelConfig("remote_gpu", "qwen3:4b-instruct-2507-q4_K_M"))
    with pytest.raises(AssertionError): p.validate_http(model_request(p, **changes))
