namespace BookRent.Orchestrator.Api;

public static class OrchestratorEndpoints
{
    public static void MapOrchestratorEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder identityGroup = app.MapGroup("/api/v1/catalog");

        identityGroup.MapPost("/removeFromCatalog/{bookId:guid}", EmployeeOrchestrationSagas.RemoveBookFromCatalogSaga);
    }  
    
    
    
}