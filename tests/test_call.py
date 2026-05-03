from alerts.notifier import Notifier

notifier = Notifier()
notifier.reset_cooldowns()

# Test dry run call
print("Testing EMERGENCY call (dry run):")
result = notifier.call_owner(
    threat_score = 95,
    flags        = ["AIMING_AT_CAMERA", "weapon:gun", "time:late_night"]
)
print(f"Call result: {result}")

# Test full escalation chain
print("\nTesting full EMERGENCY chain:")
notifier.reset_cooldowns()
notifier.notify_owner(95, "EMERGENCY", ["AIMING_AT_CAMERA"])
notifier.call_owner(95, ["AIMING_AT_CAMERA", "weapon:gun"])
notifier.notify_contact(95, "EMERGENCY", ["AIMING_AT_CAMERA"])
notifier.notify_police_prompt(95, ["AIMING_AT_CAMERA", "weapon:gun"])

print("\nCall test OK")
