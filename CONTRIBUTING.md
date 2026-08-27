# Contributing Guide

## Branches
- **main** — стабильная версия проекта с проверенными и готовыми фичами.
- **dev** — основная ветка разработки.

## Workflow
1. **Как начать работу?**
   - Всегда создавайте новую ветку от `dev`.
   - Название ветки должно отражать задачу:
     - `feature/имя-фичи`
     - `bugfix/имя-багфикса`
     - `docs/обновление-документации`
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/my-new-feature
   ```
   - Делайте небольшие коммиты с понятными сообщениями на англ.

3. **Открытие Pull Request**
   - После завершения работы отправьте ветку на GitHub:
     ```bash
     git push origin feature/my-new-feature
     ```
   - Создайте Pull Request в **`dev`**.
   - В описании PR укажите, какие задачи он закрывает:
     ```
     Closes #12
     ```

4. **Ревью и слияние**
   - PR в `dev`/`main` проходит GitHub Actions CI: `ruff check`, `ruff format --check`,
     `ty check`, `pytest` (см. `.github/workflows/ci.yml`).
   - Перед пушем локально:
     ```bash
     uv sync --group dev
     uv run ruff check .
     uv run ruff format --check .
     uv run ty check
     uv run pytest
     ```
   - После зелёного CI PR ревьюится и вливается в `dev`.

5. **Обновление `main`**
   - Когда набор изменений готов к релизу, создаётся Pull Request из `dev` в `main`.
