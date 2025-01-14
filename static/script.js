$(function() {
    $("#train-algorithm-select").on("change", function() {
        const url = new URL("/train", window.location.origin);
        url.searchParams.set("algorithm", $(this).val());
        window.location.href = url.toString();  // redirect to specific algorithm to train
    });
    const $loadingOverlay = (/*html*/`
        <div class="loading-overlay">
            <div class="spinner-border text-light" role="status">
                <span class="sr-only">Loading...</span>
            </div>
            <p>Loading, please wait...</p>
        </div>
    `);

    function showLoading(id) {
        $(id).empty().append($($loadingOverlay))
    }
    function removeLoading(id) {
        $(id).find(".loading-overlay").remove()
    }
    
    function displayTrainingResults(data) {
        const $algorithmElem = $(/*html*/`
            <h4>Algorithm: ${data.algorithm_name}</h4>
        `);
        const $datasetLength = $(/*html*/`
            <p>Dataset Size: ${data.dataset.length}</p>
        `);
        const $features = $(/*html*/`
            <p>Features: [${data.features.join(", ")}]</p>
        `);
        const $targets = $(/*html*/`
            <p>Target/s: [${data.target.join(", ")}]</p>
        `);
        const hyperp = Object.entries(data.hyperparameters).filter(([k,v]) => !!v);

        const $hyperparameters = $(/*html*/`
            <div class="my-2 d-flex justify-content-between column-gap-2 row-gap-2">
                <p>Hyperparameters used:</p>
                ${hyperp.map(([k, v]) => {
                    return "<p>" + k + " = " + v + "</p>"
                }).join("")}
            </div>
        `);
        const $evaluation = $(/*html*/`
            <div class="my-2 d-flex justify-content-between column-gap-2 row-gap-2">
                ${data.evaluation.entries().map(([k, v]) => {
                    return "<h5>" + k.split('_').join(' ').toUpperCase() + "</h5><p>" + v.toString() + "</p>"
                }).join("")}
            </div>
        `)
        const $plots = $(/*html*/`
            <div class="my-2">
                ${data.plots.entries().map(([k, v]) => {
                    return '<div><h5>' + k.split('_').join(' ').toUpperCase() + '</h5><img src="data:image/png;base64,' + v + '" />' + '</div>'
                }).join("")}
            </div>
        `)
    }

    $("form#training-form").on("submit", function(event) {
        event.preventDefault();
        const formData = new FormData();
        const fileInput = $(this).find("input[type=file][name=training_data]");

        if (fileInput[0].files.length === 0) {
            alert('Please select a csv file for the dataset.');
            return;
        }
        const file = fileInput[0].files[0];
        formData.set('training_data', file, file.name);
        const features = $(this).find("input[name=features]").val().split(",").map(v => v?.trim()).filter(v => !!v);
        const targets = $(this).find("input[name=targets]").val().split(",").map(v => v?.trim()).filter(v => !!v);
        formData.append('features', JSON.stringify(features))
        formData.append('target', JSON.stringify(targets))
        const algo = $(this).find("select[name=algorithm]").val();
        formData.set("algo", algo)
        let hyperparameters = Object.fromEntries(new FormData($(this).get(0)));
        delete hyperparameters["training_data"]
        delete hyperparameters["algorithm"]
        delete hyperparameters["features"]
        delete hyperparameters["targets"]
        hyperparameters = Object.fromEntries(Object.entries(hyperparameters).map(([key, value]) => value === "" || (value !== 0 && value !== "0" && !value) ? [key, null] : (
            !Number.isNaN(Number.parseFloat(value)) ? [key, Number.parseFloat(value)] : (
                value === "true" || value === "false" ? [key, value === "true"] : [key, value]
            )
        )))
        formData.set("hyperparameters", JSON.stringify(hyperparameters))

        const xhr = new XMLHttpRequest();
        const url = new URL("/api/train", window.location.origin)
        xhr.open('POST', url.toString(), true);

        xhr.onload = function() {
            removeLoading("#training-results");
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    console.log('Response:', response);
                } catch (e) {
                    console.log('Error parsing JSON:', e);
                }
            } else {
                console.log('Error uploading file: ' + xhr.statusText);
            }
        };
        showLoading("#training-results");
        xhr.send(formData);
    })


    function fetchTrainingHistory() {
        if (window.location.pathname === "/train") {
            const url = new URL("/api/train_history", window.location.origin)
            showLoading("#training-results");
            $.get(url.toString())
                .done(function(data) {
                    removeLoading("#training-results");
                    console.log("FETCHED HISTORY:", data)
                    if (data.error) {
                        console.log(data.error)
                    } else{
                        displayTrainingResults(data)
                    }
                })
                .fail(function(_, statusText) {
                    removeLoading("#training-results");
                    console.log("ERROR FETCH:", statusText)
                })
        }
    }
    fetchTrainingHistory()
})

// <div class="col-md-4 mb-3">
// <label for="feature1" class="form-label">Feature 1</label>
// <input type="text" class="form-control" id="feature1" name="feature1" placeholder="Enter value for Feature 1" required>
// <span> : </span>
// <input type="text" class="form-control" id="feature1" name="feature1" placeholder="Enter value for Feature 1" required>
// </div>