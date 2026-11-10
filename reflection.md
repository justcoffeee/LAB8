# Reflection cá nhân - cần sinh viên viết

W8 mục 24 không cho AI thay reflection. Viết **300-500 từ** bằng lời của mình:

> Sau LAB07 đã biết xây dựng Transformer. Sau LAB08, tại sao thực tế thường
> không bắt đầu bằng train Transformer từ đầu?

Bắt buộc đề cập data, pretrained representation, computational cost, downstream
task, fine-tuning và limitation/domain shift. Trích số liệu thật từ results.csv
và domain_results.json; phân biệt observation -> evidence -> interpretation.

## Dàn ý hỗ trợ

1. Bối cảnh LAB7 và bài toán IMDb, giải thích pretraining/fine-tuning bằng lời
   của bạn; phân biệt tokenizer đã học với encoder random initialization.
2. Đối chiếu đường 10/25/50/100% của cả hai model, số liệu frozen/full tuning;
   nêu giới hạn do cùng epoch nhưng khác số optimizer step.
3. So sánh chi phí downstream với chi phí pretraining lịch sử không đo ở lab;
   xem T64/128/256 có đáng chi phí tăng thêm trong kết quả của bạn không.
4. Phân tích domain shift IMDb -> Yelp, một lỗi tự tin cao, các giới hạn về
   truncation/tokenization/labels. Không suy từ accuracy thành hiểu ngôn ngữ.
5. Hướng thực nghiệm tiếp theo và cầu nối sequence classification -> token
   classification/NER ở LAB09.

## Q1-Q6 bắt buộc (sinh viên tự trả lời)

1. Model nào hưởng lợi hơn khi ít training data? Dùng cả đường cong, không chỉ 10%.
2. Nếu pretrained tốt hơn scratch, kết quả chứng minh điều gì và chưa chứng minh gì?
3. Nếu frozen gần bằng fine-tuned, điều đó có thể có ý nghĩa gì?
4. Nếu fine-tuning hơn frozen nhiều, điều đó gợi ý điều gì? Có confound từ LR không?
5. Nếu performance giảm trên domain khác, những nguyên nhân nào cần kiểm tra?
6. Nếu tăng length gần như không đổi performance nhưng tốn thời gian, có nên tăng tiếp?

## Individual learning check

Tự trả lời 12 câu W8 mục 27: pretraining/fine-tuning; MLM; tokenizer; subword;
CLS; attention mask; frozen encoder; tham số thay đổi khi fine-tuning; dữ liệu
downstream nhỏ; domain shift; chi phí sequence length; confidence/correctness.

Nội dung reflection cá nhân: **chưa được cung cấp**. Không coi dàn ý này là bài
reflection 300-500 từ đã hoàn thành.
