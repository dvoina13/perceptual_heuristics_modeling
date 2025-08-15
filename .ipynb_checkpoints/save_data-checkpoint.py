import numpy as np
import torch



def save_data_(seed, p, mlp_head, valid_acc, valid_roc_score, valid_loss_arr, final_val_acc, final_score, pred, y, batchsize, layer_to_use, last_layer_dim):

    torch.save(mlp_head.state_dict(), "models_saved/mlp_model_batch_size"+ str(batchsize) + "_layer_" +  str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".pt")

    np.save("results/valid_acc_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_" + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", np.array(valid_acc))
    np.save("results/valid_roc_score_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_" + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", np.array(valid_roc_score))
    np.save("results/valid_loss_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_" + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", np.array(valid_loss_arr))
    
    np.save("results/mlp_head_weights_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_" + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", mlp_head[1].weight.detach().numpy())
    np.save("results/mlp_head_bias_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_" + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", mlp_head[1].bias.detach().numpy())
    
    np.save("results/after_silencing_final_val_acc_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_"  + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", final_val_acc)
    np.save("results/after_silencing_final_score_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_"  + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", final_score)
    np.save("results/after_silencing_PRED_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_"  + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", pred.detach().numpy())
    np.save("results/after_silencing_Y_seed_" + str(seed) + "_p_" + str(p) + "_batch_size_"  + str(batchsize) + "_layer_" + str(layer_to_use) + "_last_layer_" + str(last_layer_dim) + ".npy", y.detach().numpy())



