import os
import subprocess
import shutil
from os import path
import json
import requests, time

def setup_conf():
    
    if not path.exists('./env'): os.makedirs('./env')
    if not path.exists('./conf'): os.makedirs('./conf')
    
    for file in os.listdir('./env.example'):
        
        if not path.exists(f'./env/{file}'):
            
            if path.isfile(f'./env.example/{file}'):
                shutil.copyfile(f'./env.example/{file}', f'./env/{file}')
            
            if path.isdir(f'./env.example/{file}'):
                shutil.copytree(f'./env.example/{file}', f'./env/{file}')
            
    for file in os.listdir('./conf.example'):
        
        if not path.exists(f'./conf/{file}'):
            
            if path.isfile(f'./conf.example/{file}'):
                shutil.copyfile(f'./conf.example/{file}', f'./conf/{file}')
            
            if path.isdir(f'./conf.example/{file}'):
                shutil.copytree(f'./conf.example/{file}', f'./conf/{file}')
                
    if not path.exists(f'./up-service.json'):
        
        shutil.copyfile(f'./up-service.example.json', f'./up-service.json')
                
def setup_redash():
    
    with open('./docker_files/redash/package.json', 'r+') as f:
        
        packages = json.load(f)
        
        if 'node' in packages['engines']:
            del packages['engines']['node']
        
        f.seek(0)
        f.truncate()
        
        f.write(json.dumps(packages, indent=2))
        
    if not path.exists('./docker_files/redash/client/dist'):
        
        subprocess.run(['cd ./docker_files/redash; yarn --frozen-lockfile; yarn build'], shell=True)
        
def setup_kafka_cluster():
    
    if path.exists('./ssl/kafka/truststore/ca-key'): return
    
    if path.exists('./ssl/kafka/keystore'):
        shutil.rmtree('./ssl/kafka/keystore')
    
    if path.exists('./ssl/kafka/truststore'):
        shutil.rmtree('./ssl/kafka/truststore')
    
    if path.exists('./ssl/kafka/cert-file'):
        os.remove('./ssl/kafka/cert-file')
    
    if path.exists('./ssl/kafka/cert-signed'):
        os.remove('./ssl/kafka/cert-signed')
        
    if path.exists('./ssl/kafka/ca-cert'):
        os.remove('./ssl/kafka/ca-cert')

    subprocess.run(['cd ssl/kafka; bash ./generate_certs.sh; cd ../../'], shell=True)

    if path.exists('ssl/kafka/secrets'):
        shutil.rmtree('ssl/kafka/secrets')

    os.makedirs('./ssl/kafka/secrets')
    
    shutil.copyfile('./ssl/kafka/creds', './ssl/kafka/secrets/creds')
    shutil.copyfile('./ssl/kafka/keystore/kafka.keystore.jks', './ssl/kafka/secrets/kafka.keystore.jks')
    shutil.copyfile('./ssl/kafka/truststore/kafka.truststore.jks', './ssl/kafka/secrets/kafka.truststore.jks')

def setup_clickhouse_cluster():
    
    if path.exists('./ssl/clickhouse/ca-cert.pem'): return
    
    subprocess.run(['cd ssl/clickhouse; bash ./generate_certs.sh; cd ../../'], shell=True)

def setup_conductor():
    
    if path.exists('docker_files/conductor_docker/server/config'):
        shutil.rmtree('docker_files/conductor_docker/server/config')
    
    if path.exists('docker_files/conductor_docker/serverAndUI/config'):
        shutil.rmtree('docker_files/conductor_docker/serverAndUI/config')

    shutil.copytree('./conf/conductor/server/', 'docker_files/conductor_docker/server/config')
    shutil.copytree('./conf/conductor/server-and-ui/', 'docker_files/conductor_docker/serverAndUI/config')
   
def concat_env_file():
    
    subprocess.run(["""awk 'FNR==1{print ""}1' env/.env.* > .env"""], shell=True)
    
    if not path.exists('./tls/temporal/.pki'):
        os.makedirs('./tls/temporal/.pki')
        
    if not path.exists('./tls/temporal/pki'):
        os.makedirs('./tls/temporal/pki')

class DockerCompose:

    def __init__(self, services):
        self.services = services
        self.__docker_compose_files = []
    
    def add_docker_compose(self, service, docker_compose_file):
        if service in self.services and self.services[service]:
            self.__docker_compose_files.append(docker_compose_file)
    
    def get_docker_compose_files(self):
        return self.__docker_compose_files
        
def run_all_services(up_services):

    compose = DockerCompose(up_services)
    
    # subprocess.run([
    #     "DOCKER_BUILDKIT=1 sudo docker build -t temporal_tls:test -f ./docker_files/temporal/Dockerfile.tls . --network=host && " +
    #     'docker run --rm -v temporal_tls_pki:/pki -v ${PWD}/tls/temporal/.pki:/pki-out temporal_tls:test',
    # ], shell=True)
    compose.add_docker_compose('network', '1.docker-compose.network.infra.yml')
    compose.add_docker_compose('minio-cluster', '2.docker-compose.minio-cluster.infra.yml')
    # compose.add_docker_compose('nocodb', '3.docker-compose.nocodb.infra.yml')
    compose.add_docker_compose('kafka-cluster', '3.docker-compose.kafka.infra.yml')
    # compose.add_docker_compose('miniflux', '5.docker-compose.miniflux.infra.yml')
    # compose.add_docker_compose('debezium', '6.docker-compose.debezium.infra.yml')
    # compose.add_docker_compose('temporal', '7.docker-compose.temporal.infra.yml')
    # compose.add_docker_compose('clickhouse', '9.docker-compose.clickhouse.infra.yml')
    # compose.add_docker_compose('redash', '10.docker-compose.redash.infra.yml')
    # compose.add_docker_compose('rebula-graph', '11.docker-compose.rebula-graph.infra.yml')
    # compose.add_docker_compose('redis', '14.docker-compose.redis.infra.yml')
    # compose.add_docker_compose('directus', '12.docker-compose.directus.infra.yml')
    # compose.add_docker_compose('cvat', '13.docker-compose.cvat.infra.yml')
    # compose.add_docker_compose('budibase', '15.docker-compose.budibase.infra.yml')
    
    print(compose.get_docker_compose_files())

    subprocess.run([
        f"docker-compose {' '.join(map(lambda x: f'-f {x}', compose.get_docker_compose_files()))} up -d"
    ], shell=True)

def connect_debezium_and_kafka(retries=0):
    
    subprocess.run(['pip install requests'], shell=True)
    
    try:
        data = None 
        
        with open('./conf/debezium_conf.json') as f:
            data = json.load(f)

        result = requests.post(
            url='http://127.0.0.1:8087/connectors/',
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            data=json.dumps(data)
        )
        
        if result.status_code not in [200, 201]:

            result_2 = requests.put(
                url=f'http://127.0.0.1:8087/connectors/{data["name"]}/config',
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                data=json.dumps(data['config'])
            )
    except:
        
        if retries < 3:
            
            time.sleep(5)
            
            retries += 1
            
            connect_debezium_and_kafka(retries)
        
def post_setup():
    
    subprocess.run([
        "sudo find ./ssl/clickhouse -type d -exec chmod 755 {} \;"
    ], shell=True)
    
    subprocess.run([
        "sudo find ./ssl/clickhouse -type f -exec chmod 755 {} \;"
    ], shell=True)
    
if __name__ == '__main__':
    
    setup_conf()
    
    up_services = None 
    
    with open('./up-service.json') as f:
        
        up_services = json.load(f)
    
    # setup_redash()
            
    # setup_kafka_cluster()
    
    # setup_clickhouse_cluster()
    
    # setup_conductor()
    
    concat_env_file()
    
    run_all_services(up_services)
    
    # post_setup()
    
    # connect_debezium_and_kafka()
