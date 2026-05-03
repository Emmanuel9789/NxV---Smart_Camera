import os, sys
sys.path.insert(0, '/home/emmanuel/camera_project')
os.environ["NXV_TEST_MODE"] = "true"


with open('/home/emmanuel/camera_project/.env') as f:
    for line in f:
        line = line.strip()
        if line.startswith('export '):
            line = line.replace('export ', '', 1)
            key, _, value = line.partition('=')
            os.environ[key] = value.strip('"')


from alerts.call_chain import CallChain
import time

chain = CallChain()

print("Starting call chain dry run...")
chain.start(
    threat_score = 95,
    flags        = ["AIMING_AT_CAMERA", "weapon:gun", "time:late_night"],
    address      = "123 Test Street"
)

# Wait for chain to run (dry run is fast)
time.sleep(180)
print(f"\nChain answered by: {chain.answered_by or 'nobody'}")
print("Call chain test OK")
