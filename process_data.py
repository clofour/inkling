from shared import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, CATEGORIES, IMAGE_SIZE
import os.path as path
import json
import numpy as np
import cairocffi as cairo

def read_ndjson(path):
    data = []

    with open(path, "r") as file:
        for line in file.readlines():
            line_data = json.loads(line)
            drawing = line_data["drawing"]

            data.append(drawing)

    return data

def preprocess_data(data):
    preprocessed_data = []

    for drawing in data:
        strokes = []
        for stroke in drawing:
            x = np.array(stroke[0], dtype=np.float32)
            y = np.array(stroke[1], dtype=np.float32)

            strokes.append(np.vstack([x, y]))

        preprocessed_data.append(strokes)

    return preprocessed_data

# Source: https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
def process_data(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images

def postprocess_data(data):
    return np.array(data)


for category in CATEGORIES:
    raw_category_file_path = path.join(RAW_DATA_DIR, f"{category}.ndjson")
    processed_category_file_path = path.join(PROCESSED_DATA_DIR, category)

    if not path.exists(f"{processed_category_file_path}.npy"):
        print(f"Processing {category} data")

    
        data = read_ndjson(raw_category_file_path)
        preprocessed_data = preprocess_data(data)
        processed_data = process_data(preprocessed_data, side=IMAGE_SIZE)
        postprocessed_data = np.array(processed_data)
        np.save(processed_category_file_path, postprocessed_data)
        