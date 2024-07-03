import torch

def policy_string_analyze(policy_precision_string):
    policy_precision = []
    for char in policy_precision_string:
        if char == '0':
            policy_precision.append(torch.float16)
        elif char == '1':
            policy_precision.append(torch.float32)
    return policy_precision