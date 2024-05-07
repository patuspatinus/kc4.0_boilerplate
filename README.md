# Chạy các services 
Vaò thư mục deploy/dev

sudo python docker-up-all.py 

# Lựa chọn các services sẽ chạy
Xem trong file up-service.json, giá trị true là service sẽ chạy và false sẽ không chạy

# Chỉnh sửa các biến môi trường
Vào thư mục env, chỉnh sửa theo từng tệp

# Tắt hết các service
sudo python docker-down-all.py

# Vào localhost:8082 để xem dữ liệu kafka với tk admin/pass
# Vào localhost:9011 để xem file với tk minioadmin/minioadmin

# Workflow
- Giả lập 1 event dummy gửi tới kafka (event chứa đường dẫn url của ảnh )
- Check event, kéo url ảnh từ datalake minio
- Xử lý ảnh
- Gửi lại ảnh về minio, bắn trả lại event tới kafka

# Minio
- fput: Đẩy ảnh lên minio
- fget: Kéo ảnh từ minio về

# Kafka
- Producer: Tạo gửi event
- Consumer: Nhận event

# Giả lập event 
- Tạo 1 bucket chứa ảnh ví dụ trên datalake để kéo về
- Tạo 1 bucket chứa ảnh output của mô hình để đẩy lên