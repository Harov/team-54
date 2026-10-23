# Connect free training compute

Prepared October 23, 2026. The website continues using Cloudflare until Render is connected. The current ridge regression uses CPU calculations; a GPU is not required.

## 1. Put the upload package on GitHub

1. Download and unzip `render-compute-upload.zip` from this chat. It contains a `worker` folder with the training service. The package contains no passwords, connection strings, or installed dependencies.
2. Create a GitHub repository. A private repository is suitable if you grant Render access to that repository.
3. Use **Add file → Upload files**, drag in the extracted **worker folder**, and commit the files. Keep the folder structure: GitHub should show `worker/package.json`.

## 2. Create the Render service

In [Render](https://dashboard.render.com/), choose **New → Web Service**, connect GitHub, and select that repository.

| Setting | Value |
| --- | --- |
| Runtime | Node |
| Root directory | `worker` |
| Build command | `npm ci --include=dev && npm run build:render` |
| Start command | `npm run start:render` |
| Instance type | Free |
| Health check path | `/health` |

If Free is not offered for your account, stop before selecting a paid instance. Use a supported Node LTS release, at least 22.12.

## 3. Add environment variables in Render

| Name | Value |
| --- | --- |
| `MONGODB_URI` | Your existing Atlas driver connection string, including its database user and password |
| `MONGODB_DATABASE` | `cuda_research`, matching this website's database |
| `RENDER_TRAINING_KEY` | A random shared key, at least 32 characters |

On your Mac, this command generates a random key and copies it to your clipboard without displaying it:

```sh
openssl rand -hex 32 | pbcopy
```

Paste the key into Render's environment settings. Save the same value in a password manager so you can add it to Cloudflare. Do not paste it into chat or commit it to GitHub.

Add the service's displayed outbound IP addresses to your MongoDB Atlas network access list. Use those addresses rather than opening the database to every IP.

Deploy. After the service starts, its public URL will look like `https://your-service.onrender.com`. Visiting that URL with `/health` appended should return `{"status":"ok"}`. This checks the HTTP service; the first research run also checks database access.

## 4. Connect Cloudflare to Render

In the `cuda-research` Worker's secret settings, add:

| Secret | Value |
| --- | --- |
| `RENDER_TRAINING_URL` | The public HTTPS `onrender.com` origin, without `/health` or `/train` |
| `RENDER_TRAINING_KEY` | Exactly the same shared key you saved in Render |

Open the website's **Connect data → Where your model runs**, then choose **Refresh configuration**. It should show Render. Start one research run and confirm forecasts can still be read from MongoDB.

You can share the public Render service URL in chat for help checking the deployment. Keep keys and database passwords in the services' secret settings.

## Costs and limits

[Render's free documentation](https://render.com/docs/free) lists 750 instance hours per workspace per calendar month. Idle services sleep after 15 minutes and can take about a minute to wake. Free services have temporary local files; this project saves results in MongoDB. Render describes free instances as intended for hobby and testing use, not production workloads. Bandwidth and build limits also apply; review the dashboard before enabling paid options.

[Google Colab](https://research.google.com/colaboratory/faq.html) provides free interactive notebook compute, sometimes including GPUs, subject to availability. It is useful for manual experiments; it is not a continuously available backend for this site.

The app never automatically repeats a timed-out training POST or starts a second model on another provider. The original job may still save its results. Check saved forecasts before retrying.
