import subprocess

def down_all_container():
    
    subprocess.run(["sudo docker-compose $(find *.docker-compose.*.infra.yml | sed -e 's/^/-f /') down"], shell=True)

down_all_container()
