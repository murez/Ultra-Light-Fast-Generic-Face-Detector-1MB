from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
import cv2
def detect_face(orig_image, net_type='slim', candidate_size=750, test_device='cuda', threshold=0.6):
    if net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    elif args.net_type == 'RFB':
        model_path = "models/pretrained/version-RFB-320.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    return predictor.predict(image, candidate_size, threshold)