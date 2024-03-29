# The challengers team -- VAIPE competition

**1. Hướng dẫn test (inference)**

B1. clone repo tại: git@github.com:khaidoandk97/The_challengers_team_VAIPE.git (nhánh master)

B2. Tải folder "trained_models" tại **https://drive.google.com/file/d/1uOGQ_eLay1MYYzwj24JB7SX3pqawYkZO/view?usp=sharing** giải nén và đặt vào WORKSPACE (cùng cấp với folder lib, run.py)

B3. Tạo docker image:

    docker build -t [image name] .
    vd: 
    docker build -t vaipe-the_challengers_team .
    
B4. Tạo các volume: data (mount tới folder chứa dữ liệu test), output (folder lưu kết quả model sinh ra) 

    docker volume create --opt device=[absolute path to test data on host device] --opt type=none --opt o=bind  output
    docker volume create --opt device=[absolute path to saved folder on host device] --opt type=none --opt o=bind  data

    vd:
       docker volume create --opt device=/home/user/Desktop/output --opt type=none --opt o=bind  output
       docker volume create --opt device=/home/user/Desktop/VAIPE/document/dataset/public_test_new --opt type=none --opt o=bind  data

B5. chạy container

    docker run -it --name [container name] -v data:/app/data -v output:/app/output [image name]
    vd: docker run -it --name containter -v data:/app/data -v output:/app/output vaipe-the_challengers_team

khi chạy xong kết quả sẽ được sinh ra file "results.csv" để  trong thư mục trên máy host mount tới volum data của docker


**2. Hướng dẫn train:**

B1. train detection engine.

    train model "Cascade R-CNN" sử dụng repo mmdetection: https://github.com/open-mmlab/mmdetection

    sử dụng config mặc định "cascade_rcnn_r50_fpn_1x_coco.py" https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py để train model

B2. OCR Prescription engine.

    Sử dụng prertain vietocr - backbone: VGG19-bn - Transformer https://github.com/pbcquoc/vietocr

B3. classification engine.

    Model do nhóm tự thiết kế
    
    train theo hướng dẫn trên nhánh "classification_engine": https://github.com/khaidoandk97/The_challengers_team_VAIPE/tree/classification_engine
    
