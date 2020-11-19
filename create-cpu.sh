gcloud beta compute --project=steady-scope-295806 instances create gan-dense-1 --zone=us-central1-a --machine-type=n2-custom-8-65536\
    --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE --preemptible\
    --service-account=431972424519-compute@developer.gserviceaccount.com\
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --tags=http-server,https-server \
    --image-family=tf2-ent-latest-cpu-ubuntu-1804 --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB --boot-disk-type=pd-standard --boot-disk-device-name=gan-dense-1 \
    --metadata-from-file startup-script=./start.sh \
    --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
    #--metadata-from-file shutdown-script=./powerdown.sh \