namespace BookRent.Orchestrator.Api.Requests;

public class RemoveFromCatalogRequest
{
    public Guid BookId { get; set; }
    public Guid UserId { get; set; }
}