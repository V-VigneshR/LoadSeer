import threading
import time
from aws_manager import *
from config import *

class LoadBalancer:
    def __init__(self, monitor):
        self.monitor = monitor
        self.switching_event = threading.Event()

    def warmup_and_switch(self, target_group_to_use):
        """Handle instance warmup and switching"""
        if self.switching_event.is_set():
            return

        self.switching_event.set()
        print(f"[JMeter-ML] Switching to: {target_group_to_use}")

        try:
            manage_instance('start', target_group_to_use)
            register_instances(target_group_to_use)

            start_time = time.time()
            while time.time() - start_time < 60:
                response = elbv2.describe_target_health(
                    TargetGroupArn=TARGET_GROUPS[target_group_to_use],
                    Targets=[{'Id': iid,'Port':8080} for iid in INSTANCE_IDS[target_group_to_use]]
                )
                if response['TargetHealthDescriptions'][0]['TargetHealth']['State'] == 'healthy':
                    deregister_instances(self.monitor.current_state)
                    manage_instance('stop', self.monitor.current_state)
                    self.monitor.current_state = target_group_to_use
                    print(f"[SUCCESS] Switched to {target_group_to_use}")
                    return
                time.sleep(5)

            print(f"Health check timeout for {target_group_to_use}")

        except Exception as e:
            print(f"Transition failed: {str(e)}")
            if target_group_to_use != 'small':
                self.warmup_and_switch('small')
        finally:
            self.switching_event.clear()

    def background_scaler(self):
        """Background scaling thread"""
        while True:
            try:
                target_state = self.monitor.choose_load_balancer_ml()
                if target_state and target_state != self.monitor.current_state and not self.switching_event.is_set():
                    threading.Thread(target=self.warmup_and_switch, args=(target_state,)).start()
                time.sleep(5)
            except Exception as e:
                print(f"Background scaler error: {str(e)}")
                time.sleep(10)  # Longer sleep on error to prevent spam