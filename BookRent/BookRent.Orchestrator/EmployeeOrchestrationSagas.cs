using BookRent.Orchestrator.Clients;

namespace BookRent.Orchestrator;

internal static class EmployeeOrchestrationSagas
{

    public static IResult RemoveBookFromCatalogSaga()
    {
        return Results.Ok();
    }
    
    
}