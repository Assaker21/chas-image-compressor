import cv2
import numpy as np
import struct
import os

MIN_BLOCK_SIZE = 4
THRESHOLD = 10.0

def needed_bytes(x):
    if x == 0:
        return 1
    nbits = x.bit_length()
    nbytes = (nbits + 7) // 8
    return min(nbytes, 8)

def write_int(f, val, nbytes):
    f.write(val.to_bytes(nbytes, byteorder='little', signed=False))

def read_int(data, pos, nbytes):
    val = int.from_bytes(data[pos:pos+nbytes], byteorder='little', signed=False)
    return val, pos + nbytes


def compute_average_color_and_mse(image, x, y, block_size):
    h, w = image.shape[:2]
    x_end = min(x + block_size, h)
    y_end = min(y + block_size, w)

    block = image[x:x_end, y:y_end]

    avg_b = np.mean(block[:, :, 0])
    avg_g = np.mean(block[:, :, 1])
    avg_r = np.mean(block[:, :, 2])
    avg_color = (avg_b, avg_g, avg_r)

    diff = block.astype(np.float32) - np.array(avg_color, dtype=np.float32)
    mse = np.mean(diff ** 2)

    return avg_color, mse


def subdivide_block(image, x, y, block_size, threshold, blocks, min_block_size=MIN_BLOCK_SIZE):
    avg_color, mse = compute_average_color_and_mse(image, x, y, block_size)
    
    if mse > threshold and block_size > min_block_size:
        half = block_size // 2
        subdivide_block(image, x,     y,     half, threshold, blocks, min_block_size)
        subdivide_block(image, x,     y+half, half, threshold, blocks, min_block_size)
        subdivide_block(image, x+half, y,     half, threshold, blocks, min_block_size)
        subdivide_block(image, x+half, y+half, half, threshold, blocks, min_block_size)
    else:
        R = int(round(avg_color[2]))
        G = int(round(avg_color[1]))
        B = int(round(avg_color[0]))
        blocks.append((x, y, block_size, R, G, B))


def compress_image_to_chas(input_path, chas_path, threshold=THRESHOLD, min_block_size=MIN_BLOCK_SIZE):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read input image: {input_path}")

    h, w = image.shape[:2]

    blocks = []
    for x in range(0, h, 16):
        for y in range(0, w, 16):
            block_size = min(16, min(h - x, w - y))
            subdivide_block(image, x, y, block_size, threshold, blocks, min_block_size)

    blocks.sort(key=lambda b: (b[0], b[1]))

    if not blocks:
        raise ValueError("No blocks found (image might be empty).")

    encoded_blocks = []
    prev_size, prev_r, prev_g, prev_b = blocks[0][2], blocks[0][3], blocks[0][4], blocks[0][5]
    run_positions = [(blocks[0][0], blocks[0][1])]

    for i in range(1, len(blocks)):
        x, y, size, r, g, b = blocks[i]
        if (size == prev_size) and (r == prev_r) and (g == prev_g) and (b == prev_b):
            run_positions.append((x, y))
        else:
            encoded_blocks.append((prev_size, prev_r, prev_g, prev_b, run_positions))
            prev_size, prev_r, prev_g, prev_b = size, r, g, b
            run_positions = [(x, y)]
    encoded_blocks.append((prev_size, prev_r, prev_g, prev_b, run_positions))

    number_of_runs = len(encoded_blocks)

    max_w = w
    max_h = h
    max_num_runs = number_of_runs

    max_size = 0
    max_r = 0
    max_g = 0
    max_b = 0
    max_run_length = 0
    max_x = 0
    max_y = 0

    for (size, R, G, B, positions) in encoded_blocks:
        if size > max_size: max_size = size
        if R > max_r:       max_r = R
        if G > max_g:       max_g = G
        if B > max_b:       max_b = B
        run_len = len(positions)
        if run_len > max_run_length:
            max_run_length = run_len
        for (xx, yy) in positions:
            if xx > max_x: max_x = xx
            if yy > max_y: max_y = yy

    w_nbytes       = needed_bytes(max_w)
    h_nbytes       = needed_bytes(max_h)
    num_runs_nbytes= needed_bytes(max_num_runs)
    size_nbytes    = needed_bytes(max_size)
    r_nbytes       = needed_bytes(max_r)
    g_nbytes       = needed_bytes(max_g)
    b_nbytes       = needed_bytes(max_b)
    runlen_nbytes = needed_bytes(max_run_length)
    x_nbytes       = needed_bytes(max_x)
    y_nbytes       = needed_bytes(max_y)

    with open(chas_path, "wb") as f:
        f.write(struct.pack("10B",
                            w_nbytes,
                            h_nbytes,
                            num_runs_nbytes,
                            size_nbytes,
                            r_nbytes,
                            g_nbytes,
                            b_nbytes,
                            runlen_nbytes,
                            x_nbytes,
                            y_nbytes))

        write_int(f, w, w_nbytes)
        write_int(f, h, h_nbytes)

        write_int(f, number_of_runs, num_runs_nbytes)

        for (size, R, G, B, positions) in encoded_blocks:
            write_int(f, size, size_nbytes)
            write_int(f, R,    r_nbytes)
            write_int(f, G,    g_nbytes)
            write_int(f, B,    b_nbytes)
            run_length = len(positions)
            write_int(f, run_length, runlen_nbytes)

            for (xx, yy) in positions:
                write_int(f, xx, x_nbytes)
                write_int(f, yy, y_nbytes)


def decompress_chas_to_image(chas_path, output_path):
    with open(chas_path, "rb") as f:
        data = f.read()

    pos = 0

    (
        w_nbytes,
        h_nbytes,
        num_runs_nbytes,
        size_nbytes,
        r_nbytes,
        g_nbytes,
        b_nbytes,
        runlen_nbytes,
        x_nbytes,
        y_nbytes
    ) = struct.unpack_from("10B", data, pos)
    pos += 10

    w, pos = read_int(data, pos, w_nbytes)
    h, pos = read_int(data, pos, h_nbytes)

    num_runs, pos = read_int(data, pos, num_runs_nbytes)

    reconstructed = np.zeros((h, w, 3), dtype=np.uint8)

    for _ in range(num_runs):
        size, pos = read_int(data, pos, size_nbytes)
        R,    pos = read_int(data, pos, r_nbytes)
        G,    pos = read_int(data, pos, g_nbytes)
        B,    pos = read_int(data, pos, b_nbytes)
        run_length, pos = read_int(data, pos, runlen_nbytes)

        positions = []
        for __ in range(run_length):
            xx, pos = read_int(data, pos, x_nbytes)
            yy, pos = read_int(data, pos, y_nbytes)
            positions.append((xx, yy))

        for (xx, yy) in positions:
            x_end = min(xx + size, h)
            y_end = min(yy + size, w)
            reconstructed[xx:x_end, yy:y_end] = (B, G, R)

    cv2.imwrite(output_path, reconstructed)

input_file = "files/input.png"
input_raw_file = "files/input.raw"
compressed_file = "files/output.chas"
output_file = "files/output.png"


min_block_sizes = [1, 2, 4]
thresholds = [10, 50, 500, 1000, 10000]

for min_block_size in min_block_sizes:
    for threshold in thresholds:
        
        compressed_file = f"files/output_{min_block_size}x{min_block_size}_{threshold}.chas"
        output_file = f"files/output_{min_block_size}x{min_block_size}_{threshold}.png"
        
        compress_image_to_chas(input_file, compressed_file, threshold=threshold, min_block_size=min_block_size)
        decompress_chas_to_image(compressed_file, output_file)

        original_image = cv2.imread(input_file)
            
        final_image = cv2.imread(output_file)

        print(f"Min block size: {min_block_size}x{min_block_size}")
        print(f"Threshold: {threshold}")
        psnr = cv2.PSNR(original_image, final_image)
        print(f"PSNR: {psnr:.3f}dB")

        compressed_size = os.path.getsize(compressed_file)
        compression_ratio = original_image.nbytes / compressed_size
        print(f"Compression ratio: {compression_ratio:.3f}")

        q = 0.8
        s = (2 * q * psnr * (1 - q) * compression_ratio) / (q * psnr + (1 - q) * compression_ratio)
        print("Harmonic Mean: ", s)
        print("\n\n")

with open("files/input.raw", "wb") as file:
    file.write(original_image.tobytes())
