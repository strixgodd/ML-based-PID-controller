# ================================================================
# Embedded ML-based Adaptive Cruise Control (ACC)
# Modified version that replaces classical PID with trained ML model
# Using TensorFlow Lite for inference on Raspberry Pi
# ================================================================

from gpiozero import Motor, PWMOutputDevice, DistanceSensor, Button
from time import sleep
import math
import numpy as np
import tensorflow as tf   # ==== MODIFIED: TensorFlow Lite used for ML inference ====

# ==== MODIFIED: Load the TensorFlow Lite model ====
interpreter = tf.lite.Interpreter(model_path="pid_mimic_model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === GPIO and Constants ===
ENA = 17
IN1 = 27
IN2 = 22
IN3 = 23
IN4 = 24
ENB = 25
TRIG = 13
ECHO = 6
ENCODER_DT = 5
COUNTS_PER_REV = 400
WHEEL_DIAMETER = 6.5  # cm
WHEEL_CIRCUM = math.pi * WHEEL_DIAMETER
SAMPLE_INTERVAL = 0.08  # seconds

# PID and control (partially used — Kp, Ki, Kd also used for pid_output computation)
Kp = 4.5
Ki = 1.5
Kd = 0.2
alpha = 0.97
distance_filter_alpha = 0.25
setVelocity = 25
headwayTime = 1.0
standstillDistance = 15
pwmStepLimit = 20
minPWM = 30
deadbandError = 1
maxTargetVelocityDecrease = 1.5
maxIntegral = 100
MAX_ACCEL = 10  # cm/s^2

# ==== NEW: StandardScaler parameters (from your Jupyter training) ====
# NOTE: Replace these example values with actual scaler.mean_ and scaler.scale_ values
VE_mean, VE_std =  1.02941176, 2.60144523          # velocity_error mean and std
DE_mean, DE_std = 117.44669118, 80.79292765          # distance_error mean and std
PID_mean, PID_std = 27.03125, 32.24425589        # pid_output mean and std
# (These placeholders must be updated with real numbers from training)

# === State Variables ===
pulse_count = 0
measuredVelocity = 0
filteredDistance = None
prevError = 0
integral = 0
lastPwmOutput = 0
lastTargetVelocity = setVelocity

# === Hardware setup ===
def encoder_callback():
    global pulse_count
    pulse_count += 1

encoder = Button(ENCODER_DT, pull_up=True)
encoder.when_pressed = encoder_callback
distance_sensor = DistanceSensor(echo=ECHO, trigger=TRIG, max_distance=4, threshold_distance=0.3)
motor_left = Motor(forward=IN1, backward=IN2, pwm=True)
motor_right = Motor(backward=IN3, forward=IN4, pwm=True)
pwm_ena = PWMOutputDevice(ENA)
pwm_enb = PWMOutputDevice(ENB)

def ewma_filter(prev, new_sample, alpha):
    return alpha * prev + (1 - alpha) * new_sample

# === Main Control Loop ===
def main():
    global pulse_count, measuredVelocity, prevError, integral, \
        lastPwmOutput, lastTargetVelocity, filteredDistance, setVelocity

    try:
        while True:
            # --- Ultrasonic Distance Reading (EWMA) ---
            rawDistance = int(round(distance_sensor.distance * 100))
            if filteredDistance is None:
                filteredDistance = rawDistance
            else:
                filteredDistance = int(round(
                    ewma_filter(filteredDistance, rawDistance, distance_filter_alpha)
                ))

            # --- Encoder Velocity Calculation (EWMA) ---
            count = pulse_count
            pulse_count = 0
            revs = count / float(COUNTS_PER_REV or 1)
            dist = revs * WHEEL_CIRCUM
            rawSpeed = dist / SAMPLE_INTERVAL
            measuredVelocity = int(round(
                ewma_filter(measuredVelocity, rawSpeed, alpha)
            ))

            # --- Safety Distance & Target Velocity ---
            safeDistance = int(round(standstillDistance + headwayTime * measuredVelocity))
            distanceError = int(round(rawDistance - safeDistance))
            v_sq = measuredVelocity ** 2 + 2 * MAX_ACCEL * distanceError
            if v_sq <= 0:
                targetVelocity = 0
            else:
                targetVelocity = int(round(math.sqrt(v_sq)))
            targetVelocity = min(targetVelocity, setVelocity)

            if filteredDistance <= standstillDistance:
                targetVelocity = 0
                integral = 0

            if targetVelocity < lastTargetVelocity:
                diff = lastTargetVelocity - targetVelocity
                if diff > maxTargetVelocityDecrease:
                    targetVelocity = lastTargetVelocity - int(round(maxTargetVelocityDecrease))
            lastTargetVelocity = targetVelocity

            velocityError = targetVelocity - measuredVelocity

            # ============================================================
            # ==== MODIFIED SECTION: Compute pid_output + ML model =======
            # ============================================================

            # ==== NEW: Compute pid_output (3rd input feature from training) ====
            derivative = (velocityError - prevError) / SAMPLE_INTERVAL
            integral += velocityError * SAMPLE_INTERVAL
            integral = max(-maxIntegral, min(integral, maxIntegral))
            pid_output = Kp * velocityError + Ki * integral + Kd * derivative
            prevError = velocityError

            # ==== NEW: Apply same normalization as training ====
            scaled_input = np.array([[
                (velocityError - VE_mean) / VE_std,
                (distanceError - DE_mean) / DE_std,
                (pid_output - PID_mean) / PID_std
            ]], dtype=np.float32)

            # ==== ML model inference ====
            interpreter.set_tensor(input_details[0]['index'], scaled_input)
            interpreter.invoke()
            pred_pwm = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # ==== PWM output processing ====
            pwmOutput = int(np.clip(pred_pwm, 0, 255))

            if pwmOutput > lastPwmOutput + pwmStepLimit:
                pwmOutput = lastPwmOutput + pwmStepLimit
            elif pwmOutput < lastPwmOutput - pwmStepLimit:
                pwmOutput = lastPwmOutput - pwmStepLimit
            lastPwmOutput = pwmOutput

            # ============================================================
            # ==== END OF ML-BASED CONTROL SECTION =======================
            # ============================================================

            pwm_val = max(0.0, min(pwmOutput / 255.0, 1.0))
            pwm_ena.value = pwm_val
            pwm_enb.value = pwm_val
            if pwm_val > 0:
                motor_left.forward()
                motor_right.forward()
            else:
                motor_left.stop()
                motor_right.stop()

            # Print debug/log values
            data_str = (
                f"{filteredDistance},{safeDistance},{distanceError},"
                f"{targetVelocity},{measuredVelocity},{velocityError},"
                f"{int(round(pid_output))},{int(round(pred_pwm))},{pwmOutput}"
            )
            print(data_str)

            sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        pwm_ena.off()
        pwm_enb.off()
        motor_left.stop()
        motor_right.stop()
        print("All systems safely shut down.")

if __name__ == "__main__":
    main()
