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
