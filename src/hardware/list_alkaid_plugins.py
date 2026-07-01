from alkaid.converter import get_available_plugins


def check_plugins():
    print("--- ALKAID PLUGIN REGISTRY ---")
    try:
        plugins = get_available_plugins()

        if not plugins:
            print("[WARNING] The registry is entirely empty. No plugins discovered.")
            return

        hgq_present = False

        for name, ep in plugins.items():
            # Extract standard entry point metadata
            group = getattr(ep, "group", "unknown_group")
            value = getattr(ep, "value", str(ep))

            print(f"[{name}] -> {value} (Group: {group})")

            # Check for HGQ in either the key or the module path
            if "hgq" in name.lower() or "hgq" in value.lower():
                hgq_present = True

        print("------------------------------\n")
        print("--- DIAGNOSTIC RESULT ---")
        if hgq_present:
            print(
                "[SUCCESS] HGQ is successfully registered and visible to Alkaid's compiler."
            )
        else:
            print("[FATAL] HGQ is missing. Alkaid cannot see the translation plugin.")

    except Exception as e:
        print(f"[ERROR] Failed to query the plugin registry: {e}")


if __name__ == "__main__":
    check_plugins()
