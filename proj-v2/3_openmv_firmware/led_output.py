"""
LED Control for Nicla Voice
Provides RGB LED control for prediction output
"""

try:
    import board
    import neopixel
    HAS_NEOPIXEL = True
except ImportError:
    HAS_NEOPIXEL = False


class LEDController:
    """Control RGB LED on Nicla Voice"""
    
    def __init__(self, pin=board.D10, num_leds=1, brightness=0.5):
        """
        Initialize LED controller
        
        Args:
            pin: GPIO pin for NeoPixel
            num_leds: Number of LEDs (Nicla Voice has 1)
            brightness: Brightness level (0.0-1.0)
        """
        self.brightness = brightness
        self.num_leds = num_leds
        
        if HAS_NEOPIXEL:
            try:
                self.pixels = neopixel.NeoPixel(pin, num_leds, brightness=brightness)
            except Exception as e:
                print(f"Warning: Could not initialize NeoPixel: {e}")
                self.pixels = None
        else:
            self.pixels = None
    
    def set_color(self, rgb):
        """
        Set LED to RGB color
        
        Args:
            rgb: tuple of (R, G, B) values (0-255 each)
        """
        if self.pixels is None:
            # Fallback: print to serial
            r, g, b = rgb
            print(f"[LED] RGB({r}, {g}, {b})")
            return
        
        try:
            r, g, b = rgb
            # Apply brightness
            r = int(r * self.brightness)
            g = int(g * self.brightness)
            b = int(b * self.brightness)
            self.pixels.fill((r, g, b))
        except Exception as e:
            print(f"Error setting LED color: {e}")
    
    def turn_off(self):
        """Turn off LED"""
        self.set_color((0, 0, 0))
    
    def pulse(self, rgb, speed=100):
        """
        Pulse LED with given color
        
        Args:
            rgb: (R, G, B) color
            speed: pulse speed in ms per step
        """
        import time
        for i in range(100):
            brightness = abs(50 - i) / 50.0
            r, g, b = rgb
            r = int(r * brightness)
            g = int(g * brightness)
            b = int(b * brightness)
            self.set_color((r, g, b))
            time.sleep(speed / 1000.0)
    
    def blink(self, rgb, times=3, duration=200):
        """
        Blink LED
        
        Args:
            rgb: (R, G, B) color
            times: number of blinks
            duration: on-time in ms
        """
        import time
        for _ in range(times):
            self.set_color(rgb)
            time.sleep(duration / 1000.0)
            self.turn_off()
            time.sleep(duration / 1000.0)


class SerialOutput:
    """Format and output results to serial monitor"""
    
    @staticmethod
    def print_prediction(label, confidence, all_probs, index_to_label):
        """Print prediction with all probabilities"""
        print(f"\n{'='*50}")
        print(f"PREDICTION: {label.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*50}")
        print("All class probabilities:")
        for idx in range(len(index_to_label)):
            class_label = index_to_label[idx]
            prob = all_probs[idx]
            # Visual bar
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {class_label:12} | {bar} | {prob:.4f}")
        print()
    
    @staticmethod
    def print_error(error_msg):
        """Print error message"""
        print(f"\n[ERROR] {error_msg}\n")
    
    @staticmethod
    def print_status(status_msg):
        """Print status message"""
        print(f"[INFO] {status_msg}")
