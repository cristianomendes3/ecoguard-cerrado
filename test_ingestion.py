from src.data_ingestion import DataIngestionPipeline

def testar_ingestao():
    print("--- Iniciando Teste de Ingestão ---")
    pipeline = DataIngestionPipeline()

    # Consultar alertas simulados
    alertas = pipeline.query_mapbiomas_alerts(("2026-05-01", "2026-05-31"))

    if alertas:
        alerta = alertas[0]
        print(f"Alerta encontrado: {alerta['id']}")

        # Buscar item STAC
        item = pipeline.fetch_sentinel_item(alerta['bbox'], "2026-05-01/2026-06-01")

        if item:
            # Extrair tensor
            tensor = pipeline.extract_spectral_tensor(item, alerta['bbox'], bands=['blue', 'green', 'red', 'nir'])
            print(f"Sucesso! Dimensões do Tensor: {tensor.shape}")
        else:
            print("Falha: Item STAC não encontrado.")
    else:
        print("Falha: Nenhum alerta MapBiomas retornado.")

if __name__ == "__main__":
    testar_ingestao()