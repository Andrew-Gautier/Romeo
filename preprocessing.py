import numpy as np
def pad_sequences(sequences, maxlen, padding='post', value=0):
    """Pads sequences to the same length."""
    output = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Truncate
            new_seq = seq[:maxlen]
        else:
            # Pad
            pad_length = maxlen - len(seq)
            if padding == 'post':
                new_seq = seq + [value] * pad_length
            else:  # 'pre'
                new_seq = [value] * pad_length + seq
        output.append(new_seq)
    return np.array(output)

print("Constants and padding function defined successfully!")