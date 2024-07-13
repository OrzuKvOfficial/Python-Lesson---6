import bluetooth

target_name = "My Phone"
target_address = None

nearby_devices = bluetooth.discover_devices()

for address in nearby_devices:
    if target_name == bluetooth.lookup_name(address):
        target_address = address
        break

if target_address is not None:
    print("Found target bluetooth device with address", target_address)
else:
    print("Could not find target bluetooth device nearby")
