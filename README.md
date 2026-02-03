docker build -t rightmoveacrclp.azurecr.io/clip-worker:latest .

docker push rightmoveacrclp.azurecr.io/clip-worker:latest


az eventgrid event-subscription create \
    --name rightmove-queue-sub \
    --source-resource-id /subscriptions/b2dc8e41-a4c4-4f97-a443-7446dfe9dce2/resourceGroups/rightmove-rg/providers/Microsoft.Storage/storageAccounts/rightmoveukstorage \
    --endpoint-type azurestoragequeue \
    --endpoint https://rightmoveukstorage.queue.core.windows.net/rightmove-images-queue \
    --included-event-types Microsoft.Storage.BlobCreated \
    --subject-begins-with "/blobServices/default/containers/*/*_rightmove_images/"


az containerapp env var set \
  --name clip-worker-app \
  --resource-group rightmove-rg \
  --variables AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=rightmoveukstorage;AccountKey="key";EndpointSuffix=core.windows.net"


az containerapp update \
  --name clip-worker-app \
  --resource-group rightmove-rg \
  --set-env-vars AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=rightmoveukstorage;AccountKey="key";EndpointSuffix=core.windows.net" \
INPUT_DIR="/data/input" \
OUTPUT_DIR="/data/output" \
BATCH_SIZE="32"


az storage queue list \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --output table


az storage container list \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --output table


az storage blob upload \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --container-name london-rent \
  --file ./1.jpg \
  --name test-1.jpg

az storage message put \
  --content '{"url":"https://rightmoveukstorage.blob.core.windows.net/london-rent/01-02-2026/rent_london_rightmove_images/107278532/1.jpg"}' \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key"


az storage blob exists \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --container-name london-rent \
  --name "01-02-2026/rent_london_rightmove_images/107278532/1.jpg"


az storage blob upload \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --container-name london-rent \
  --file ./1.jpg \
  --name test-1.jpg



FOLDER_PREFIX="01-02-2026/rent_london_rightmove_images/107278532/"

for blob in $(az storage blob list \
    --account-name rightmoveukstorage \
    --account-key "key" \
    --container-name london-rent \
    --prefix "$FOLDER_PREFIX" \
    --query "[].name" \
    --output tsv)
do
  az storage message put \
    --content "{\"url\":\"https://rightmoveukstorage.blob.core.windows.net/london-rent/$blob\"}" \
    --queue-name rightmove-images-queue \
    --account-name rightmoveukstorage \
    --account-key "key"
  echo "Enqueued: $blob"
done


az storage queue clear \
  --name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key"


az storage message put   --content '{"url":"https://rightmoveukstorage.blob.core.windows.net/london-rent/01-02-2026/rent_london_rightmove_images/107278532/1.jpg"}'   --queue-name rightmove-images-queue   --account-name rightmoveukstorage   --account-key "key"

az storage message put   --content '{"url":"https://rightmoveukstorage.blob.core.windows.net/london-rent/1.jpg"}'   --queue-name rightmove-images-queue   --account-name rightmoveukstorage   --account-key "key"


az storage message put \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --content '{
    "container": "london-rent",
    "prefix": "01-02-2026/rent_london_rightmove_images/107278532/"
  }'

az storage message put \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --content '{
    "container": "london-rent",
    "prefix": "01-02-2026/rent_london_rightmove_images/113354258/"
  }'


az storage message put \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --content '{
    "container": "london-rent",
    "prefix": "01-02-2026/rent_london_rightmove_images/107278532/1.jpg"
  }'



az storage message put \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --content '{
    "container": "london-rent",
    "prefix": "1.jpg"
  }'



# Delete the queue
az storage queue delete --name rightmove-images-queue --account-name rightmoveukstorage --account-key "key"

# Recreate it
az storage queue create --name rightmove-images-queue --account-name rightmoveukstorage --account-key "key"


az storage queue list --account-name rightmoveukstorage --account-key "key"


az storage message put \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key "key" \
  --content '{
    "url": "https://rightmoveukstorage.blob.core.windows.net/london-rent/1.jpg"
  }'

docker run -e AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=rightmoveukstorage;AccountKey="key";EndpointSuffix=core.windows.net" clip-worker

# Delete all messages in the queue
az storage message clear \
  --queue-name rightmove-images-queue \
  --account-name rightmoveukstorage \
  --account-key key


## stop a job

az storage blob delete \
  --container-name london-rent \
  --name "01-02-2026/rent_london_rightmove_images/121284986/1.json" \
  --account-name rightmoveukstorage \
  --account-key key

## clear messages
az storage message clear   --queue-name rightmove-images-queue   --account-name rightmoveukstorage   --account-key key


## update the container app

az containerapp update   --name clip-worker-app   --resource-group rightmove-rg   --image rightmoveacrclp.azurecr.io/clip-worker:latest   --set-env-vars AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=rightmoveukstorage;AccountKey=key INPUT_DIR="/data/input" OUTPUT_DIR="/data/output" BATCH_SIZE="32"

## tail last contain app logs 

az containerapp logs show \
  --name clip-worker-app \
  --resource-group rightmove-rg \
  --tail 50

## delete json

az storage blob delete \
  --container-name london-rent \
  --name 01-02-2026/rent_london_rightmove_images/121284986/2.json \
  --account-name rightmoveukstorage

## delete all jons in folder 

az storage blob delete-batch \
  --account-name rightmoveukstorage \
  --source london-rent \
  --pattern "01-02-2026/rent_london_rightmove_images/121284986/*.json" \
  --account-key key

az storage blob delete-batch \
  --account-name rightmoveukstorage \
  --source london-rent \
  --pattern "01-02-2026/rent_london_rightmove_images/124719503/*.json" \
  --account-key key


## start  job

az storage message put   --queue-name rightmove-images-queue   --account-name rightmoveukstorage   --account-key key   --content '{
    "container": "london-rent",
    "prefix": "01-02-2026/rent_london_rightmove_images/124719503/"
  }'

az storage message put   --queue-name rightmove-images-queue   --account-name rightmoveukstorage   --account-key key   --content '{
    "container": "london-rent",
    "prefix": ""
  }'


  ✅ What you should do:

Wait for 0000005 to finish activating.

Once active, check that replicas are running (5 in your case).

Monitor the queue logs — processing should speed up.

You can force it active via CLI if needed:

az containerapp revision activate \
    --name clip-worker-app \
    --resource-group rightmove-rg \
    --revision clip-worker-app--0000005


## check app container updates 

az containerapp logs show --name clip-worker-app --resource-group rightmove-rg --follow

