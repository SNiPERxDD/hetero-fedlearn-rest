#!/usr/bin/env python3
"""Diagnostic script to verify LAN IP detection and UDP beacon configuration."""

import sys
from worker.worker_dfs import get_all_lan_ips, beacon_targets, get_lan_ip
from master.master_dfs import get_lan_ip as master_get_lan_ip

def test_lan_detection():
    """Test and display LAN IP detection results."""
    
    print("=" * 70)
    print("LAN DETECTION DIAGNOSTIC")
    print("=" * 70)
    
    # Test worker-side detection
    print("\n[WORKER] Enumerated LAN IPs:")
    all_ips = get_all_lan_ips()
    for ip in all_ips:
        print(f"  - {ip}")
    if not all_ips:
        print("  WARNING: No non-loopback IPs detected!")
    
    print("\n[WORKER] Selected primary LAN IP:")
    primary_ip = get_lan_ip()
    print(f"  {primary_ip}")
    if primary_ip == "127.0.0.1":
        print("  WARNING: Fell back to localhost! Auto-discovery may not work on same network.")
    
    # Test beacon targets
    print("\n[WORKER] UDP Beacon targets for auto-discovery:")
    targets = beacon_targets(primary_ip)
    for target in targets:
        print(f"  - {target}")
    
    # Test master-side detection
    print("\n[MASTER] Master LAN IP:")
    master_ip = master_get_lan_ip()
    print(f"  {master_ip}")
    if master_ip == "127.0.0.1":
        print("  WARNING: Master fell back to localhost!")
    
    # Cross-machine registration recommendation
    print("\n" + "=" * 70)
    print("CROSS-MACHINE SETUP RECOMMENDATION")
    print("=" * 70)
    
    if primary_ip != "127.0.0.1":
        print(f"\nFor Windows worker {primary_ip}:5000 to register with macOS master {master_ip}:18080:")
        print(f"\n  python3 start_worker.py --allow-unsupported-python \\")
        print(f"    --master-endpoint http://{master_ip}:18080 \\")
        print(f"    --advertised-endpoint http://{primary_ip}:5000")
    else:
        print("\nLAN IP detection returned 127.0.0.1 on both ends.")
        print("This suggests network interface enumeration is failing.")
        print("\nTroubleshooting steps:")
        print("  1. Check that network interfaces are active: 'ip addr show' or 'ipconfig /all'")
        print("  2. Try probing external addresses: tests if routing interface can be found")
        print("  3. Check with: python3 -c \"import socket; print(socket.gethostname(), socket.gethostbyname(socket.gethostname()))\"")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        test_lan_detection()
    except Exception as e:
        print(f"Error during diagnostic: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
