# app.py
from flask import Flask, request, render_template, flash, redirect, url_for, send_file
import os
import csv
import uuid
from ocr_extract import extract_text_from_image     # Phase 1
from field_extractor import extract_invoice_fields  # Phase 2

app = Flask(__name__)
app.secret_key = "your-secret-key"

@app.route("/", methods=["GET", "POST"])
def index():
    all_results = []
    csv_filename = None

    if request.method == "POST":
        images = request.files.getlist("image")
        if not images:
            return "No files uploaded.", 400

        os.makedirs("uploads", exist_ok=True)
        batch_id = uuid.uuid4().hex
        csv_filename = f"invoices_{batch_id}.csv"
        csv_path = os.path.join("uploads", csv_filename)

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

            for image in images:
                image_path = os.path.join("uploads", image.filename)
                image.save(image_path)

                raw_text = extract_text_from_image(image_path)
                result = extract_invoice_fields(raw_text)

                all_results.append({"filename": image.filename, "fields": result})

                writer.writerow([
                    image.filename,
                    result.get("Invoice No", ""),
                    result.get("Date", ""),
                    result.get("Total Amount", ""),
                    result.get("Vendor", "")
                ])

                os.remove(image_path)

        return render_template("index.html", all_results=all_results, csv_filename=csv_filename)

    return render_template("index.html")

@app.route("/download/<filename>")
def download_csv(filename):
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found.", 404

@app.route("/append-to-existing", methods=["POST"])
def append_to_existing_csv():
    csv_filename = request.form.get("csv_filename")
    new_csv_path = os.path.join("uploads", csv_filename)
    existing_csv_path = os.path.join("uploads", "invoices.csv")

    if not os.path.exists(new_csv_path):
        flash("No generated CSV to append.")
        return redirect(url_for("index"))

    # Create existing file if it doesn't exist
    if not os.path.exists(existing_csv_path):
        with open(existing_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

    # Append data from new CSV to existing CSV (skipping header)
    with open(new_csv_path, 'r') as new_file, open(existing_csv_path, 'a', newline='') as existing_file:
        reader = csv.reader(new_file)
        writer = csv.writer(existing_file)
        next(reader)  # skip header
        for row in reader:
            writer.writerow(row)

    flash("✅ Appended to invoices.csv successfully!")
    return send_file(existing_csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)