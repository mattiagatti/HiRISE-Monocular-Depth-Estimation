import os
from pathlib import Path
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from glpdepth.model import GLPDepth
import open3d as o3d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, H = 512, 512


def load_model(path):
    model = GLPDepth(max_depth=700.0).to(device)
    weights = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
    weights = {k[6:]:v for (k, v) in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    return model


def generate_mesh(dtm, image, image_path):
    # prepare points
    points = np.zeros(shape=(W * H, 3), dtype="float32")
    colors = np.zeros(shape=(W * H, 3), dtype="float32")
    for i in range(H):
        for j in range(W):
            points[i * H + j, 0] = abs(j - 512)
            points[i * H + j, 1] = i
            points[i * H + j, 2] = dtm[i, j]
            colors[i * H + j, :3] = image[i, j]
    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7, n_threads=1)[0]
    # mesh transformations
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    # remove weird artifacts
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # save mesh
    out_path = f"{image_path.stem}.obj"
    o3d.io.write_triangle_mesh(filename=out_path, mesh=mesh)
    return out_path


def predict(image_path):
    image_path = Path(image_path)
    pil_image = Image.open(image_path).convert("L")
    # transform image to torch
    to_tensor = transforms.ToTensor()
    torch_image = to_tensor(pil_image).to(device).unsqueeze(0)
    # model predict
    with torch.no_grad():
        pred_dtm = model(torch_image)
    # transform torch to numpy
    pred_dtm = pred_dtm.squeeze().cpu().detach().numpy()
    pred_dtm = pred_dtm.max() - pred_dtm
    # create 3d model
    image_scaled = np.asarray(pil_image) / 255.0
    obj_path = generate_mesh(pred_dtm, image_scaled, image_path)

    # return correct image
    fig, ax = plt.subplots()
    im = ax.imshow(pred_dtm, cmap="jet", vmin=0, vmax=np.max(pred_dtm))
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return [data, obj_path]


pretrained_path = Path("pretrained/best_model.ckpt")
model = load_model(pretrained_path)

title = "Mars DTM Estimation"
description = "This demo predicts a DTM from an image of the martian surface. Then, by using a surface reconstruction " \
                "algorithm, the 3D model is generated and it can also be downloaded."
examples = [[f"examples/{name}", 3] for name in sorted(os.listdir("examples"))]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Input Image", height=512, width=512, sources=["upload"])
    ],
    outputs=[
        gr.Image(label="DTM"),
        gr.Model3D(label="3D Model", clear_color=[0.0, 0.0, 0.0, 0.0])
    ],
    examples=examples,
    allow_flagging="never",
    cache_examples=False,
    title=title,
    description=description
).launch(server_name="0.0.0.0", server_port=8080)