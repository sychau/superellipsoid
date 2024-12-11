import trimesh
import torch
import numpy as np

from neural_network import NetworkParameters
from loss_function import euclidean_dual_loss, overlapping_loss
from sampler import EqualDistanceSamplerSQ
from visualization import save_prediction_as_ply, visualize_voxels, visualize_mesh

def main():
    # Change parameters here
    dataset_path = "./data"
    model_name = "dog"
    num_epochs = 2000
    num_primitives = 24
    num_samples = 500
    # Threshold of probability to discard, higher is more selective
    prob_threshold = 0.7
    # color of the target point cloud in visualization
    ground_truth_color = [0, 255, 255] 
    # visualize_voxels(f"{dataset_path}/{model_name}/voxel_and_sdf.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loading data
    surface_points = trimesh.load(f"{dataset_path}/{model_name}/surface_points.ply", process=False)
    surface_points_tensor = torch.tensor(surface_points.vertices, dtype=torch.float32)
    surface_points_tensor = surface_points_tensor.unsqueeze(0).cuda()

    voxels = np.load(f"{dataset_path}/{model_name}/voxel_and_sdf.npz")["voxels"]
    voxels_tensor = torch.tensor(voxels, dtype=torch.float32)
    voxels_tensor = voxels_tensor.unsqueeze(0).unsqueeze(0).cuda() # (1, 1, 64, 64, 64)

    # Create model
    network_params = NetworkParameters(
        "octnet", n_primitives=num_primitives,
        use_sq=True, make_dense=True,
        use_deformations=False,
        train_with_bernoulli=True
    )
    model = network_params.network(network_params)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training iteration
    for i in range(num_epochs):
        optimizer.zero_grad()

        # Calling forward()
        superquadraic_params = model(voxels_tensor) # PrimitiveParameters
        prediction = superquadraic_params.members # (prob, translation, rotations, sizes, shapes, deformation)
        prob, translation, rotations, sizes, shapes, deformation = prediction
        # Compute loss, modify the dictionary to change the loss function behaviour
        loss, debug_stats = euclidean_dual_loss(prediction, surface_points_tensor, {
                "regularizer_type": [
                    "parsimony_regularizer",
                    "sparsity_regularizer",
                    "entropy_bernoulli_regularizer",
                    "bernoulli_regularizer",
                ],
                "bernoulli_regularizer_weight": 1.0,
                "maximum_number_of_primitives": num_primitives,
                "minimum_number_of_primitives": 1,
                "entropy_bernoulli_regularizer_weight": 10**(-3),
                "sparsity_regularizer_weight": 1.0,
                "parsimony_regularizer_weight": 10**(-3),
                "overlapping_regularizer_weight": 1.0,
                "enable_regularizer_after_epoch": 10,
                "w1":0.05,
                "w2":0.05
            },
            EqualDistanceSamplerSQ(num_samples), # number of samples
            {
                "use_sq": True,
                "use_cuboids": False,
                "use_chamfer": True,
                "loss_weights": {
                    "pcl_to_prim_weight": 1.2,
                    "prim_to_pcl_weight": 0.8,
                }
            }
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss.backward()
        optimizer.step()

        if (i == 0 or i == 399 or i == 799 or i == 1199 or i == 1599 or i == 1999):
            print(f"Iteration {i}, CD Loss: {debug_stats['chamfer_loss']}")

    # Save meshes
    output_path = "./output/result.ply"
    save_prediction_as_ply(prediction, output_path, prob_threshold)

    # Display meshes comparision
    ground_truth_point_cloud = trimesh.load(f"{dataset_path}/{model_name}/surface_points.ply", process=False)
    ground_truth_point_cloud.colors = ground_truth_color
    prediction_mesh = trimesh.load_mesh(output_path)
    scene = trimesh.Scene()
    scene.add_geometry(ground_truth_point_cloud)
    scene.add_geometry(prediction_mesh)
    scene.show()

if __name__ == "__main__":
    main()
